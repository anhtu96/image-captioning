# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:02:16 2019

@author: tungo
"""
import os
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset


class CaptioningDataset(Dataset):
    """
    Create a Dataset instance that can be used with DataLoader when training. This Dataset returns batches of images and encoded captions.
  
    Initialize with:
    - image_dir: directory containing all images
    - captions: dictionary mapping images to their raw captions (string)
    - word_to_idx: dictionary mapping captions' words to indices
  
    Return a dataset of:
    - (image, caption_encoded): tuple containing every image and its encoded caption
    """
    def __init__(self, image_dir, captions, word_to_idx):
        """
        Construct a new instance of Dataset.
        """
        self.image_dir = image_dir
        self.data = []
        self.word_to_idx = word_to_idx
        for img in captions:
            for caption in captions[img]:
                self.data.append((img, caption))
    
    def __getitem__(self, idx):
        name = os.path.join(self.image_dir, self.data[idx][0])
        caption_words = self.data[idx][1].split()
        caption_encoded = [self.word_to_idx[i] for i in caption_words]
        caption_encoded = torch.Tensor(caption_encoded)
        image = Image.open(name)
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return (image, caption_encoded)
    
    def __len__(self):
        return len(self.data)