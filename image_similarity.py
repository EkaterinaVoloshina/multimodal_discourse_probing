from PIL import Image
import requests
from io import BytesIO
from tqdm.notebook import tqdm
import csv
import os
import json

import pandas as pd
import numpy as np
import torchvision.transforms as transforms

import torch
from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from itertools import combinations

class DatasetLoader(Dataset):
    def __init__(self, filename, dir_name="images",
                 image_size=256, device="cpu", 
                 download=False, from_cache=False):
        self.device = device
        self.image_size = image_size
        self.dir_name = dir_name
        self.loader = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # scale imported image
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

        if from_cache:
            self.image_tensor, self.labels = self.load_from_cache(
                dir_name=dir_name,
                filename=filename
            )
        else:
            annotations, images = self.open_file(filename)
            self.image_tensor, self.labels = self.load_dataset(
                images=images[:1000],
                annotations=annotations,
                download=download)
            
    
    def __len__(self):
        return len(self.image_tensor)

    def __getitem__(self, idx):
        return self.image_tensor[idx], self.labels[idx]
    
    def load_from_cache(self, dir_name, filename):
        annotations = pd.read_csv(filename)
        labels = []
        image_tensor = torch.Tensor(len(annotations), 3, self.image_size, self.image_size)
        for num, image_name in enumerate(os.listdir(dir_name)):
            img = Image.open(os.path.join(dir_name, image_name)).convert('RGB')
            image = self.loader(img)
            idx = image_name.split(".")[0]
            image_tensor[num, :, :, :] = image
            label = annotations[annotations["idx"] == int(idx)].values.tolist()[0]
            labels.append(label)
        return image_tensor, labels
    
    def open_file(self, filename):
        with open(filename) as f:
            content = json.load(f)
        return (content["annotations"],
        content["images"])

    def open_and_save(self, url, filename=None, download=False):
        response = requests.get(url)
        try:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            image = self.loader(img)
            if download:
                img.save(os.path.join(self.dir_name, f"{filename}.jpg"))
            return image
        except:
            print(url)

    def find_annotation(self, idx, annotations):
        labels = []
        for annotation in annotations:
            if annotation[0]["photo_flickr_id"] == idx:
                labels.append(annotation[0]["original_text"])

        return labels

    def load_dataset(self, images, annotations, download=False):
        labels = []
        image_tensor = torch.Tensor(len(images), 3, self.image_size, self.image_size)
        for num, image in tqdm(enumerate(images)):
            idx = image["id"]
            label = self.find_annotation(idx, annotations)
            if label != []:
                img = self.open_and_save(image["url_o"], 
                                         filename=image["id"],
                                         download=download)
                if img != None: 
                    image_tensor[num, :, :, :] = img
                    labels.append([str(idx), label[0], image["album_id"]])
        
        image_tensor = image_tensor[:len(labels), :, :, :]

        if download:
            with open('labels.csv','w') as f:
                w = csv.writer(f)
                w.writerow(("idx", "annotation", "album_id"))
                for label in labels:
                    w.writerow(label)
                
        return image_tensor, labels


class EncoderModel(torch.nn.Module):
    def __init__(self, img_size, 
                pretrained_model=resnet50(pretrained=True)):
        super(EncoderModel, self).__init__()
        self.model = torch.nn.Sequential(*list(pretrained_model.children())[:-1])
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((img_size, img_size))
    
    def forward(self, x):
        x = self.model(x)
        x = self.adaptive_pool(x)
        x = x.max(-1).values.max(-1).values
        return x


def get_vectors(data, model, device="cpu"):
    images = [] # change to tensor
    annotations = [] 
    albums = []
    ids = []
    for image, (idx, annotation, album_id) in data:
        image = image.to(device)
        vector = model(image)
        images.append(vector)
        ids.append(idx)
        annotations.append(annotation)
        albums.append(album_id)
    return images, ids, annotations, albums


def confuse_captions(images, annotations, ids, album, img2txt=True):

    new_values = {}

    cos_sim = cosine_similarity(images[0].detach().cpu().numpy())

    сheck_album_id = lambda img1, img2: True if album[img1] == album[img2] else False

    
    for i, vector in enumerate(cos_sim):
        closest = np.argsort(vector)
        if closest[-1] == i:
            closest_value_idx = -2
        else:
            closest_value_idx = -1

        closest_value = closest[closest_value_idx]
        while сheck_album_id(i, closest_value) and closest_value != -1:
            closest_value_idx -= 1
            closest_value = closest[closest_value_idx]
        
        if img2txt:
            new_values[ids[0][i]] = (annotations[0][i], annotations[0][closest_value])
        else: 
            new_values[annotations[0][i]] = (ids[0][i], ids[0][closest_value])
            
    
    return new_values
