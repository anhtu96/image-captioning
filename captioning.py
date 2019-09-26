# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:26:44 2019

@author: tungo
"""
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

class Captioning(object):
    """
    A class containing model, parameters... Can be used to train and generate new captions for new images.
    """
    def __init__(self, encoder, decoder, criterion, optimizer, num_epochs, print_every):
        """
        Construct a new instance.
        
        Inputs:
        - encoder: an Encoder instance
        - decoder: a Decoder instance
        - criterion: a metric for computing loss function (e.g. torch.nn.CrossEntropyLoss())
        - optimizer: a torch optimizer (Adam, SGD...)
        - num_epochs: number of epochs
        - print_every: print loss after a number of iterations
        """
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        self.encoder = encoder.to(device=self.device)
        self.decoder = decoder.to(device=self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.print_every = print_every
            
                
    def train(self, dataloader):
        """
        Function for tranining.
        
        Input:
        - dataloader: a DataLoader
        """
        for epoch in range(self.num_epochs):
            print('Epoch %d' %epoch)
            print('--------')
            for i, (img, caption) in enumerate(dataloader):
                img, caption = img.to(device=self.device), caption.to(device=self.device, dtype=torch.long)
                caption_in = caption[:, :-1]
                caption_out = caption[:, 1:]
                self.optimizer.zero_grad()
                features = self.encoder(img)
                out_lstm = self.decoder(caption_in, features)
                loss = self.criterion(out_lstm.view(out_lstm.size(0)*out_lstm.size(1), -1), caption_out.contiguous().view(caption_out.size(0)*caption_out.size(1)))
                loss.backward()
                self.optimizer.step()
                if i % self.print_every == 0:
                    print('Iter %d: loss = %f' %(i, loss.item()))
              
              
    def generate_caption(self, image_name, word_to_idx, idx_to_word, max_length):
        """
        Generate caption for an image.
        
        Inputs:
        - image_name: full path string of image
        - word_to_idx
        - idx_to_word
        - max_length: max length of a caption
        """
        image = Image.open(image_name)
        transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = transform(image).unsqueeze(0).to(device=self.device)
        vgg_features = self.encoder(img)
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        print('Generated caption: ', self.decoder.sample(vgg_features, word_to_idx, idx_to_word, max_length, self.device))