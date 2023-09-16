# coding=utf-8
import sys
sys.path.append("../..")

from detectron2_proposal_maxnms import collate_fn, extract, NUM_OBJECTS, DIM
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
from pathlib import Path
import argparse
import json
import os


class ImageDataset(Dataset):
    def __init__(self, image_dir, captions):
        self.image_dir = image_dir
        #self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.image_path_list = list(tqdm([(image_dir.joinpath(f"{x['image']}.jpg"), idx) for idx, x in captions.items()]))
       # self.image_path_list = list(tqdm([x for x in image_dir.iterdir() if x.stem in captions]))
        self.n_images = len(self.image_path_list)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path, caption = self.image_path_list[idx]
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img,
            'caption': caption,
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--root', type=str, default='demo/')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--split', default='test')
    parser.add_argument('--captions')
    parser.add_argument('--original', default=False)
    args = parser.parse_args()
    
    with open(args.captions) as f:
        captions = json.load(f)
    
    if args.original:
        captions =  {caption:{"image": image} for image, caption in captions.items()}
        


    dset_img_dir = Path(args.root).resolve()

    dataset_name = 'vist'
    
    out_dir = Path(args.outdir).resolve()
    if not out_dir.exists():
        out_dir.mkdir()

    print('Load images from', dset_img_dir)
    

    dataset = ImageDataset(dset_img_dir, captions)
    print('# Images:', dataset.n_images)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc)
