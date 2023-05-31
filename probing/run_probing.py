import os
import json
import random
import argparse

import itertools

from volta.config import BertConfig, M3PConfig
from volta.encoders import BertForVLTasks, M3PForVLTasks, BertForVLPreTraining
from volta.train_utils import tbLogger
from volta.task_utils import LoadDatasetEval, LoadLoss, EvaluatingModel

from transformers import AutoTokenizer

import numpy as np
import json

import torch
import torch.distributed as dist
from tqdm import tqdm

from dataset_utils import ConceptCapLoaderVal


def load(
    original_data,
    shuffled_data,
    lmdb,
    shuffled_lmdb,
    model_config="ctrl_vilbert_base",
    model="pytorch_model_9.bin",
    seq_len=26,
    task="images",
    batch_size=8
):
    config = BertConfig.from_json_file(f"config/{model_config}.json")
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model, do_lower_case=config.do_lower_case)

    valid_dataset = ConceptCapLoaderVal(original_data, shuffled_data, lmdb, shuffled_lmdb, tokenizer, seq_len, task, batch_size)

    model = BertForVLPreTraining.from_pretrained(model, config=config)

    return model, valid_dataset

def evaluate(model, valid_dataset, log_file, device):
    model.eval()
    model.to(device)

    logs = []
    accuracy = []

    for batch in tqdm(valid_dataset):
        ids = batch[-1]
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch[:-1])


        input_ids, input_mask, segment_ids, lm_label_ids, is_match, \
        image_feat, image_loc, image_cls, obj_labels, obj_confs, \
        attr_labels, attr_confs, image_attrs, image_label, image_mask= batch

        batch_size = input_ids.size(0)
        prediction_scores_t, prediction_scores_v_dict, seq_relationship_score, vqa_score, \
        all_attention_mask, pooled_output = model(input_ids=input_ids, 
                                                  image_feat=image_feat, 
                                                  image_loc=image_loc, 
                                                  token_type_ids=segment_ids,      
                                                  attention_mask=input_mask, 
                                                  image_cls=image_cls,
                                                  image_attention_mask=image_mask,
                                                  image_label=image_label,
                                                  attr_labels=attr_labels,
                                                  attr_confs=attr_confs,
                                                  image_attrs=image_attrs,
                                                  masked_lm_labels=lm_label_ids,
                                                  next_sentence_label=is_match,
                                                  obj_confs=obj_confs,
                                                  obj_labels=obj_labels,
                                                  get_scores=True)
        
        seq_relationship_score = torch.nn.functional.softmax(seq_relationship_score, dim=-1)
        seq_relationship_score = seq_relationship_score.detach().cpu().numpy()
        
        is_match = is_match.detach().cpu().numpy()
        
        accuracy.append(sum(np.argmax(seq_relationship_score, axis=1) == is_match)/batch_size)

        log = [{"id": ids[i], "is_match": float(is_match[i]), "score": seq_relationship_score[i].tolist()} for i in range(batch_size)]
        logs.extend(log)

    print("Accuracy: ", sum(accuracy)/len(accuracy))

    with open(log_file, "w") as file:
        json.dump(logs, file)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model in zero-shot format.")
    parser.add_argument("--original_data_path", default="caption_valid.json")
    parser.add_argument("--shuffled_data_path", default="caption_valid_random.json")
    parser.add_argument("--orig_lmdb", default="dataset_test.lmdb")
    parser.add_argument("--shuf_lmdb", default="dataset_test.lmdb")
    parser.add_argument("--model_config", default="ctrl_vilbert_base")
    parser.add_argument("--model_path", default="pytorch_model_9.bin")
    parser.add_argument("--seq_len", type=int, default=26)
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--log_file", default="logs.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", default="images", help="Fixed modality")
    args = parser.parse_args()
    
    print("Loading model and dataset..")
    model, dataset = load(args.original_data_path, args.shuffled_data_path,
                         args.orig_lmdb, args.shuf_lmdb, args.model_config, args.model_path,
                         args.seq_len, args.task, args.batch_size)
    
    print("Evaluating a model..")
    evaluate(model, dataset, args.log_file, args.device)
    print("Done!")
    
if __name__=="__main__":
    main()