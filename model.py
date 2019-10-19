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
        
        # remove classifier layers since we need to keep spatial information (num of pixels) to use Attention over pixel locations, not classify
        modules = list(self.vgg.children())[:-1]
        self.vgg = nn.Sequential(*modules)
        for params in self.vgg.parameters():
            params.requires_grad = False


    def forward(self, x):
        out = self.vgg(x)
        return out


class Decoder(nn.Module):
    """
    Decoder with Attention.
    """
    def __init__(self, pretrained=False, embedding_matrix=None, vocab_size=0, embed_dim=256, hidden_size=256):
        """
        Construct a new Decoder.
    
        Input:
        - pretrained: if True, use weights from given embedding matrix. Default: False
        - embedding_matrix: a numpy embedding matrix
        - vocab_size: number of all words
        - embed_dim: embedding dimension
        - hidden_size: hidden size of LSTM
        """
        super(Decoder, self).__init__()
        # MLPs to generate initial hidden state and cell state
        self.init_h = nn.Linear(512, hidden_size)
        self.init_c = nn.Linear(512, hidden_size)
        
        # alignment model
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(512, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        
        # embedding and LSTM
        if pretrained == True:
            vocab_size = embedding_matrix.shape[0]
            embed_dim = embedding_matrix.shape[1]
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.embedding.load_state_dict({'weight': torch.from_numpy(embedding_matrix)})  # using weights from given embedding matrix
            for params in self.embedding.parameters():
                params.requires_grad = False  # no need to train embedding layer
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + 512, hidden_size, batch_first=True)  # hidden_size = 4096, the same as output's dimension from Encoder
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def init_hidden_state(self, enc_out):
        """
        Initialize hidden state and cell state with mean values of encoder output along pixels
        
        Input:
        - enc_out: output from the Encoder with size (batch, no. of pixels, encoder dim)
        
        Return:
        - h and c: initial hidden states and cell states
        """
        enc_out_mean = torch.mean(enc_out, dim=1)
        h = self.init_h(enc_out_mean).unsqueeze(0)
        c = self.init_c(enc_out_mean).unsqueeze(0)
        return h, c
    

    def forward(self, dec_input, h, c, enc_out):
        """
        Forward pass at 1 timestep.
        
        Inputs:
        - dec_input: input to the Decoder with size (batch)
        - h, c: hidden and cell state from previous timestep with size (1, batch, hidden size)
        - enc_out: Encoder's output (batch, no. of pixels, encoder dim)
    
        Return:
        - out: (batch, 1, hidden size)
        """
        h_t = h.permute(1, 0, 2)
        energies = self.Va(torch.tanh(self.Wa(h_t) + self.Ua(enc_out)))
        alphas = self.softmax(energies)
        context = torch.sum(alphas * enc_out, dim=1).unsqueeze(1)
        embedding = self.embedding(dec_input.unsqueeze(1))
        lstm_in = torch.cat((embedding, context), dim=-1)
        out_lstm, (h, c) = self.lstm(lstm_in, (h, c))
        out = self.linear(out_lstm)
        return out, h, c
  

    def sample(self, enc_out, word_to_idx, idx_to_word, max_length, beam_search=True, k=3, device='cpu'):
        """
        Generate captions by generating 1 word at each timestep.
    
        Inputs:
        - enc_out: encoded features of 1 image, with shape (1, hidden size)
        - word_to_idx: dictionary
        - idx_to_word: list containing all words
        - max_length: max length of caption
        - beam_search: whether to use Beam search or Greedy search, default: True
        - device: a string. device = 'cuda' or 'cpu'
    
        Return:
        - captions: generated caption in string format
        """
        start = torch.tensor(word_to_idx['<START>']).to(device=device, dtype=torch.long).unsqueeze(0)
    
        # captions will have length of "max_length", with '<START>' as 1st letter
        captions = []
        with torch.no_grad():
            enc_out = enc_out.view(enc_out.size(0), enc_out.size(1), -1)
            enc_out = enc_out.permute(0, 2, 1)
            h, c = self.init_hidden_state(enc_out)
            dec_input = start
            
            # Greedy Search
            if beam_search == False or k == 1:
                captions = ['<START>']
                for i in range(max_length-1):
                    out, h, c = self.forward(dec_input, h, c, enc_out)
                    _, predict = torch.max(out, -1)
                    next = predict.clone().cpu().numpy()[0][0]  # get string of next word
                    if idx_to_word[next] == '<END>':
                        break
                    captions.append(idx_to_word[next])
                    dec_input = predict.squeeze(1)
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
                        out, h, c = self.forward(dec_input, h, c, enc_out)
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
                                word_idx = torch.tensor(word_to_idx[row['caption'][-1]]).unsqueeze(0).to(device=device)
                                out, row['h'], row['c'] = self.forward(word_idx, row['h'], row['c'], enc_out)
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