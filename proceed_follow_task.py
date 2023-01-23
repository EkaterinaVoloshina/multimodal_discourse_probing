from PIL import Image
import requests
from io import BytesIO
from tqdm.notebook import tqdm
import csv
import os
import json
import random

import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from sklearn.metrics.pairwise import cosine_similarity

from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from itertools import combinations

class DatasetLoader:
    def __init__(self, filename, dir_name="images",
                 image_size=256, device="cpu", 
                 download=False, from_cache=False):
        self.device = device
        self.filename = filename
        self.image_size = image_size
        self.dir_name = dir_name
        self.loader = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # scale imported image
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])   
    
    def open_file(self):
        with open(self.filename) as f:
            content = json.load(f)
        return (content["annotations"],
        content["images"])

    def open_and_save(self, images, download=False):
        images_id = {}
        for image in images[:100]:
            response = requests.get(image["url_o"])
            try:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                image_tensor = self.loader(img)
                if download:
                    img.save(os.path.join(self.dir_name, f"{image['id']}.jpg"))
                images_id[image["id"]] = image_tensor
            except:
                print(image["url_o"])
        return images_id 

    def load_from_cache(self, dir_name):
        images_id = {}
        for num, image_name in enumerate(os.listdir(dir_name)):
            img = Image.open(os.path.join(dir_name, image_name)).convert('RGB')
            image = self.loader(img)
            idx = image_name.split(".")[0]
            images_id[idx] = image
        return images_id

    def load_stories(self, annotations):
        stories = {}
        for annotation in annotations[:100]:
            text = annotation[0]["original_text"]
            story = annotation[0]["story_id"]
            order = annotation[0]["worker_arranged_photo_order"]
            photo_id = annotation[0]["photo_flickr_id"]
           
            if story in stories:
                stories[story][order] = [text, photo_id]
            else:
                stories[story] = {order: [text, photo_id], }
        return stories

    def load(self, from_cache=False, download=False):
        annotations, images = self.open_file() 
        stories = self.load_stories(annotations)

        if from_cache:
            images = self.load_from_cache(self.dir_name)
        else:
            images = self.open_and_save(images, download=download)
        return images, stories


class TaskGenerator(Dataset):
    def __init__(self, images_id, stories, threshold, image_size=256):
        self.images_id = images_id
        self.image_size = image_size
        self.threshold = threshold
        self.stories = stories

        self.image_tensor, self.texts, self.labels = self.generate_task(threshold)
    
    def __len__(self):
        return len(self.image_tensor)

    def __getitem__(self, idx):
        return self.image_tensor[idx], self.texts[idx], self.labels[idx]

    def generate_proceeding(self, story):
        text_id = random.randint(0, 3)
        photo_id = random.randint(text_id + 1, 4)
        image = self.images_id[story[photo_id][1]] # image tensor
        text = story[text_id][0]
        answer = 0
        return image, text, answer

    def generate_following(self, story):
        text_id = random.randint(1, 4)
        photo_id = random.randint(0, text_id - 1)
        image = self.images_id[story[photo_id][1]] # image tensor
        text = story[text_id][0]
        answer = 1
        return image, text, answer

    def generate_task(self, threshold):
        image_tensor = torch.Tensor(len(self.stories), 3, 
                                    self.image_size, self.image_size)
        texts = []
        labels = []
        for num, (story_id, story) in enumerate(self.stories.items()):
            if num < threshold:
                image, text, answer = self.generate_proceeding(story)
            else:
                image, text, answer = self.generate_following(story)
            image_tensor[num,:,:,:] = image
            texts.append(text)
            labels.append(answer)
        return image_tensor, texts, labels