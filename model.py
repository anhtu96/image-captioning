# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:06:35 2019

@author: tungo
"""
import torch
import torch.nn as nn
from torchvision import models

class Encoder(nn.Module):
    """
    CNN model for image encoding, using pretrained VGG19 network
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.vgg.classifier = self.vgg.classifier[:6]  # remove last fc layer
        for params in self.vgg.parameters():
            params.requires_grad = False


    def forward(self, x):
        out = self.vgg(x)
        return out


class Decoder(nn.Module):
    """
    Decoder uses images' features from Encoder as initial hidden states for LSTM layer, feeds embedded ground-truth captions into this LSTM layer to generate predicted captions.
    """
    def __init__(self, embedding_matrix):
        """
        Construct a new Decoder.
    
        Input:
        - embedding_matrix: a numpy embedding matrix
        """
        super(Decoder, self).__init__()
        vocab_size = embedding_matrix.shape[0]
        embed_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})  # using weights from given embedding matrix
        for params in self.embedding.parameters():
            params.requires_grad = False  # no need to train embedding layer
        self.lstm = nn.LSTM(embed_dim, 4096, batch_first=True)  # hidden_size = 4096, the same as output's dimension from Encoder
        self.linear = nn.Linear(4096, vocab_size)
    

    def forward(self, captions, features):
        """
        Forward pass.
        
        Inputs:
        - captions: tensor of encoded captions, with shape (N x T)
        - features: tensor of images' features after Encoder layer, with shape (N x H)
    
        Return:
        - tensor with shape (N x H x V)
        """
        embedding = self.embedding(captions)
        h0 = features.unsqueeze(0)  # initial hidden states = features from Encoder
        c0 = torch.zeros_like(h0)  # initial cell states = all zeros
        out_lstm, _ = self.lstm(embedding, (h0, c0))
        out = self.linear(out_lstm)
        return out
  

    def sample(self, features, word_to_idx, idx_to_word, max_length, device):
        """
        Generate captions by generating 1 word at each timestep.
    
        Inputs:
        - features: encoded features of 1 image, with shape (1 x H)
        - word_to_idx: dictionary
        - idx_to_word: list containing all words
        - max_length: max length of caption
        - device: a string. device = 'cuda' or 'cpu'
    
        Return:
        - captions: generated caption in string format
        """
        start = torch.tensor(word_to_idx['<START>']).to(device=device)
    
        # captions will have length of "max_length", with '<START>' as 1st letter
        captions = ['<START>']
        with torch.no_grad():
            x = self.embedding(start).unsqueeze(0).unsqueeze(0) # make x to have 3 dimensions
            h = features.unsqueeze(0)
            c = torch.zeros_like(h)
            for i in range(max_length):
                out_lstm, (h, c) = self.lstm(x, (h, c))
                out = self.linear(out_lstm)
                _, predict = torch.max(out, -1)
                next = predict.clone().cpu().numpy()[0][0]  # get stirng of next word
                if idx_to_word[next] == '<END>':
                    break
                captions.append(idx_to_word[next])
                x = self.embedding(predict)
            captions = ' '.join([word for word in captions if word != '<START>'])
            return captions