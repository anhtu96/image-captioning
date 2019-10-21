# Image Captioning
This is my implementation of image captioning with Attention. I use PyTorch throughout this repository since it is my favorite Deep Learning framework.

The paper which I refer to is [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) by Xu et al. Most of the concepts in this repository are taken from the paper, but there are some little differences in my implementation.

## Overview
### Dataset
Since I don't have a strong GPU for personal use, I use Google Colab. When it comes to dealing with large datasets with a huge number of files, it cannot handle so well. So instead of using big datasets like MSCOCO, I use Flickr8k dataset in this repository.

The images are first resized to smaller size of '`224 x 224` and normalized to the mean = [0.485, 0.456, 0.406], variance = [0.229, 0.224, 0.225].

The raw captions are added `<START>` and `<END>` tokens and padded to the same length with `<PAD>`. After that they are encoded to vectors of indices so we can apply algorithms on them.
### Encoder-Decoder architecture
This type of model uses an Encoder to encode the input, then feed the encoded input to a Decoder, a type of sequence model, to generate word at each timestep. In this image captioning task, the input image is encoded by a Convolutional Neural Network. I use a pretrained VGG19 and strip the last classification layers, so the Encoder output will have a dimension of `512 x 7 x 7`.

The Decoder is a Recurrent Neural Network, it can be a RNN, GRU or LSTM. In this repo I use LSTM, the same as in the paper. If we don't use Attention, the Encoder output is simply fed as the first hidden state of the Decoder's LSTM. The LSTM uses predicted word from previous timestep to generate next word.

With Attention, at each timestep, the Decoder tends to focus more on particular pixels of the Encoder output. By this, it can generate more appropriate word.
### Beam search
At decoding phase, predicted word is the one with highest probability, which is known as Greedy search. However, choosing the word with highest probability doesn't always ensure that it is the most optimal solution. For example, if the first predicted word has the highest score of `p1_1` and the next predicted word has the highest score of `p2_1`, the overall score of this sequence is `p1 = p1_1 + p2_1`. If we choose the first predicted word to have the third highest score `p1_2` and we choose the next predicted one to have the highest score `p2_2`, the overall score of this sequence is `p2 = p1_2 + p2_2`. As we can see, there are still chances that `p2` is greater than `p1`.

Unlike Greedy search, Beam search selects k best candidates at each timestep. With this technique, our Decoder can generate more complex and natural sentences (for the most time).
