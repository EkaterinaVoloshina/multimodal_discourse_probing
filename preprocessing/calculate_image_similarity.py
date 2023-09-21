import argparse
import csv
import json
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet101
from tqdm import tqdm


class DatasetLoader(Dataset):
    def __init__(self, dir_name="images", image_size=256):
        self.image_size = image_size
        self.dir_name = dir_name
        self.loader = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # scale imported image
                transforms.ToTensor(),
            ]
        )
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

        self.image_tensor, self.labels = self.load_dataset(dir_name=dir_name)

    def __len__(self):
        return len(self.image_tensor)

    def __getitem__(self, idx):
        return self.image_tensor[idx], self.labels[idx]

    def load_dataset(self, dir_name):
        image_ids = []
        image_tensor = torch.Tensor(
            len(os.listdir(dir_name)), 3, self.image_size, self.image_size
        )
        for num, image_name in enumerate(os.listdir(dir_name)):
            img = Image.open(os.path.join(dir_name, image_name)).convert("RGB")
            image = self.loader(img)
            idx = image_name.split(".")[0]
            image_tensor[num, :, :, :] = image
            image_ids.append(idx)
        return image_tensor, image_ids


class EncoderModel(torch.nn.Module):
    def __init__(self, img_size):
        super(EncoderModel, self).__init__()
        pretrained_model = resnet101(pretrained=True)
        self.model = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((img_size, img_size))

    def forward(self, x):
        x = self.model(x)
        x = self.adaptive_pool(x)
        x = x.max(-1).values.max(-1).values
        return x


def get_vectors(data, model, device="cpu"):
    image_list = []
    ids = []
    model.to(device)
    for images, idx in tqdm(data):
        images = images.to(device)
        vectors = model(images)
        image_list.append(vectors.detach().cpu().numpy())
        ids.extend(idx)

    image_list = np.concatenate(image_list)

    return image_list, ids


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the image similarity in the dataset"
    )
    parser.add_argument(
        "--dirname", default="images", help="the directory where the images are saved"
    )
    parser.add_argument(
        "--output_dir", default="vectors", help="the output directory for the vectors"
    )
    parser.add_argument(
        "--batch_size", default=8, help="the batch size of the encoder model"
    )
    parser.add_argument(
        "--image_size", default=256, help="the size of images to encode"
    )
    parser.add_argument("--device", default="cpu", help="The device to run model on")

    print("Loading data..")
    args = parser.parse_args()
    dataset = DatasetLoader(dir_name=args.dirname)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    model = EncoderModel(args.image_size)

    print("Encoding data..")
    images, ids = get_vectors(dataloader, model)

    print("Saving data..")
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    np.save(os.path.join(args.output_dir, "image_vectors.npy"), images)
    with open(os.path.join(args.output_dir, "image_ids.txt"), "w") as file:
        file.write("\n".join(ids))


if __name__ == "__main__":
    main()
