# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:00:28 2019

@author: tungo
"""
import numpy as np
import string


def get_embedding_dict(filename):
    """
    Read given embedding text file (e.g. Glove...), then output an embedding dictionary.
    
    Input:
    - filename: full path of text file in string format
    
    Return:
    - embedding_dict: dictionary contains embedding vectors (values) for all words (keys)
    """
    file = open(filename, 'r', encoding='utf8')
    embedding_dict = {}
    for line in file.read().split('\n'):
        words = line.split(' ')
        if len(words) < 2:
            continue
        embedding_dict[words[0]] = words[1:]
    return embedding_dict
 
 
def get_images_list(filename):
    """
    Return a list containing all image names from given text file.
    
    Input:
    - filename: full path of text file in string format
    
    Return:
    - images: list containing all image names
    """
    file = open(filename, 'r', encoding='utf8')
    images = [img for img in file.read().split('\n') if img != '']
    return images
    
    
def get_captions(filename, image_list):
    """
    Return a dictionary in which every image name (key) is mapped to a list of captions (value).
    For example, {'img_a': ['caption_1a', 'caption_2a'], 'img_b': ['caption_1b', caption_2b', 'caption_3b']}.
    
    Inputs:
    - filename: text file containing image names and captions
    - image_list: list of images we want to get captions
    
    Returns:
    - captions: dictionary with keys are image names and values are list of corresponding captions
    """
    file = open(filename, 'r', encoding='utf8')
    captions = {}
    table = str.maketrans('A', 'a', string.punctuation)
    for line in file.read().split('\n'):
        words = line.split()
        if len(words) < 2:
            continue
    
        # only add image from list
        if words[0][:-2] in image_list:
            if words[0][:-2] not in captions:
                captions[words[0][:-2]] = []
            # lower all captions, then add '<START>' and '<END>'
            captions[words[0][:-2]].append('<START> ' + ' '.join(words[1:]).lower().translate(table) + ' <END>')
    return captions


def get_padded_captions(captions):
    """
    Pad all captions so that they have the same length.
    
    Input:
    - captions: dictionary containing image names and list of corresponding captions
    
    Returns:
    - padded_captions: the same as captions, but with padded captions
    - max_length: length of maximum caption before padding
    """
    max_length = 0
    padded_captions = {}
    for img in captions:
        padded_captions[img] = []
    for img, caption_list in captions.items():
        for caption in caption_list:
            cap_split = caption.split()
            if len(cap_split) > max_length:
                max_length = len(cap_split)
    
    for img in padded_captions:
        padded_captions[img] = captions[img].copy()
        for i, caption in enumerate(padded_captions[img]):
            cap_split = caption.split()
            if len(cap_split) < max_length:
                padding_seq = ['<PAD>'] * (max_length - len(cap_split))
                padded_captions[img][i] +=' ' + ' '.join(padding_seq)
          
    return padded_captions, max_length


def encode_captions(captions):
    """
    Convert all captions' words into indices.
    
    Input:
    - captions: dictionary containing image names and list of corresponding captions
    
    Returns:
    - word_to_idx: dictionary of indices for all words
    - idx_to_word: list containing all words
    - vocab_size: number of words
    """
    word_counts = {}
    for name, caption_list in captions.items():
        for caption in caption_list:
            for word in caption.split():
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
    idx_to_word = ['<START>', '<END>', '<PAD>'] + [w for w in word_counts if w not in ['<START>', '<END>', '<PAD>']]
    word_to_idx = {}
    for i in range(len(idx_to_word)):
        word_to_idx[idx_to_word[i]] = i
    vocab_size = len(idx_to_word)
    return word_to_idx, idx_to_word, vocab_size


def get_embedding_matrix(idx_to_word, embedding_dict):
    """
    Get embedding matrix of all captions' words.
    
    Inputs:
    - idx_to_word: list containing all words
    - embedding_dict: dictionary of embedding vectors (e.g. Glove...)
    
    Returns:
    - embedding_matrix: matrix of all words' embedding vectors
    """
    first_embed_key = list(embedding_dict.keys())[0]
    embed_dim = len(embedding_dict[first_embed_key])
    vocab_size = len(idx_to_word)
    embedding_matrix = np.random.rand(vocab_size, embed_dim)
    for i in range(len(idx_to_word)):
        word = idx_to_word[i]
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    return embedding_matrix