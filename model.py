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
    def __init__(self, pretrained=False, embedding_matrix=None, vocab_size=0, embed_dim=0):
        """
        Construct a new Decoder.
    
        Input:
        - pretrained: if True, use weights from given embedding matrix. Default: False
        - embedding_matrix: a numpy embedding matrix
        - vocab_size: number of all words
        - embed_dim: embedding dimension
        """
        super(Decoder, self).__init__()
        if pretrained == True:
            vocab_size = embedding_matrix.shape[0]
            embed_dim = embedding_matrix.shape[1]
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embedding.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})  # using weights from given embedding matrix
            for params in self.embedding.parameters():
                params.requires_grad = False  # no need to train embedding layer
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
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
  

    def sample(self, features, word_to_idx, idx_to_word, max_length, beam_search=True, k=3, device='cpu'):
        """
        Generate captions by generating 1 word at each timestep.
    
        Inputs:
        - features: encoded features of 1 image, with shape (1 x H)
        - word_to_idx: dictionary
        - idx_to_word: list containing all words
        - max_length: max length of caption
        - beam_search: whether to use Beam search or Greedy search, default: True
        - device: a string. device = 'cuda' or 'cpu'
    
        Return:
        - captions: generated caption in string format
        """
        start = torch.tensor(word_to_idx['<START>']).to(device=device)
    
        # captions will have length of "max_length", with '<START>' as 1st letter
        captions = []
        with torch.no_grad():
            h = features.unsqueeze(0)
            c = torch.zeros_like(h)
            x = self.embedding(start).unsqueeze(0).unsqueeze(0)
            
            # Greedy Search
            if beam_search == False or k == 1:
                captions = ['<START>']
                for i in range(max_length-1):
                    out_lstm, (h, c) = self.lstm(x, (h, c))
                    out = self.linear(out_lstm)
                    _, predict = torch.max(out, -1)
                    next = predict.clone().cpu().numpy()[0][0]  # get string of next word
                    if idx_to_word[next] == '<END>':
                        break
                    captions.append(idx_to_word[next])
                    x = self.embedding(predict)
                captions = ' '.join([word for word in captions if word != '<START>'])
                return captions
        
            # Beam Search
            else:
                for i in range(k):
                    captions.append({'caption': ['<START>'], 
                                     'score': torch.Tensor([0]), 
                                     'h': h.clone(), 
                                     'c': c.clone()
                    })
                max_score = -1000
                final_caption = ''
                for i in range(max_length-1):
                    if i == 0:
                        out_lstm, (h, c) = self.lstm(x, (h, c))
                        out = self.linear(out_lstm)
                        scores, preds = torch.topk(out, k, -1)
                        scores = scores.squeeze(0).squeeze(0)
                        preds = preds.squeeze(0).squeeze(0)
                        for idx in range(len(captions)):
                            next_word_idx = preds[idx].cpu().numpy()
                            next_word = idx_to_word[next_word_idx]
                            score = torch.log(scores[idx])
                            captions[idx]['caption'].append(next_word)
                            captions[idx]['score'] = score
                            captions[idx]['h'] = h
                            captions[idx]['c'] = c
                    else:
                        score_cat, pred_cat = torch.Tensor().to(device=device), torch.Tensor().to(device=device, dtype=torch.long)
                        caption_tmp, hidden_state_tmp = [], []
                        element_to_remove = []
                        for row in captions:
                            if row['caption'][-1] == '<END>':
                                element_to_remove.append(row)
                                if max_score == -1000 or max_score < row['score'].cpu().numpy():
                                    max_score = row['score'].cpu().numpy()
                                    final_caption = ' '.join(row['caption'][1:-1])
                            else:
                                caption_tmp.append(row['caption'])
                                word_idx = torch.tensor(word_to_idx[row['caption'][-1]]).to(device=device)
                                x = self.embedding(word_idx).unsqueeze(0).unsqueeze(0)
                                out_lstm, (row['h'], row['c']) = self.lstm(x, (row['h'], row['c']))
                                out = self.linear(out_lstm)
                                hidden_state_tmp.append({'h': row['h'], 'c': row['c']})
                                scores, preds = torch.topk(out, k, -1)
                                score_cat = torch.cat((score_cat, (torch.log(scores).squeeze(0).squeeze(0) + row['score'])))
                                pred_cat = torch.cat((pred_cat, preds.squeeze(0).squeeze(0)))
                        for e in element_to_remove:
                            captions.remove(e)
                        if len(captions) == 0:
                            return final_caption
                        top_scores, top_idx = torch.topk(score_cat, k)
                        for idx in range(len(caption_tmp)):
                            row = top_idx[idx] // k
                            captions[idx]['caption'] = caption_tmp[row].copy()
                            captions[idx]['score'] = top_scores[idx].clone()
                            captions[idx]['h'] = hidden_state_tmp[row]['h'].clone()
                            captions[idx]['c'] = hidden_state_tmp[row]['c'].clone()
                            if captions[idx]['caption'][-1] != '<END>':
                                next_word_idx = pred_cat[top_idx][idx].cpu().numpy()
                                next_word = idx_to_word[next_word_idx]
                                captions[idx]['caption'].append(next_word)
                return final_caption