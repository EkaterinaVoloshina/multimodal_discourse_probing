import os
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import pickle
import numpy as np
import pandas as pd
import json
import re
import argparse

def get_embeddings(sentences, model_path="bert-base-uncased", device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    enc = tokenizer(sentences, padding=True, truncation=True,
                    max_length=512, return_tensors='pt')
    
    model.to(device)
    enc = enc.to(device)
    
    output = model(**enc, return_dict=True)
    output = output.last_hidden_state[:,0,:]
    return output.cpu().detach().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="caption_valid.json", help="the directory where the images are saved")
    parser.add_argument("--output_dir", default="vectors", help="the output directory for the vectors")
    parser.add_argument("--device", default="cpu", help="The device to run model on")
    args = parser.parse_args()
    
    print("Loading data..")
    with open(args.input_file) as file:
        captions = json.load(file)
    
    sentences = list(captions.values())
    
    print("Encoding data..")
    vectors = get_embeddings(sentences)
    
    print("Saving data..")
    np.save(os.path.join(args.output_dir, "text_vectors.npy"), vectors)
    with open(os.path.join(args.output_dir, "text_ids.txt"), "w") as file:
        file.write("\n".join(list(captions.keys())))
    
if __name__=="__main__":
    main()