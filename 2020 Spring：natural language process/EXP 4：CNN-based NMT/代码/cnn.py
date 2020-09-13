#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, embed_size=50, kernel_size=5, max_length=21, num_filter=16):
        super(CNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=embed_size, out_channels=num_filter, kernel_size=kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=max_length - kernel_size + 1)


    def forward(self, X_reshaped):
        X_conv = self.conv1d(X_reshaped)
        X_conv = F.relu(X_conv)
        X_conv_out = self.maxpool(X_conv).squeeze(-1)

        return X_conv_out

    ### END YOUR CODE

class WordCNN(nn.Module):
    
    def __init__(self, word_embed_size, kernel_size=5, num_filter=16):
        super(WordCNN, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=word_embed_size, out_channels=num_filter,
                                kernel_size=5, padding=2)
        self.active = nn.ReLU()


    def forward(self, words_embedding_reshaped):
        contex_embedding = self.conv1d(words_embedding_reshaped)
        contex_embedding = self.active(contex_embedding)

        return contex_embedding


class WordLSTM(nn.Module):
    
    def __init__(self, word_embed_size):
        super(WordLSTM, self).__init__()

        self.word_embed_size = word_embed_size
        self.lstm = nn.LSTM(word_embed_size, word_embed_size, bias=True, bidirectional=True)
        self.project = nn.Linear(2 * self.word_embed_size, self.word_embed_size, bias=False)


    def forward(self, word_embedding):
        batch_size = word_embedding.shape[1]
        
        enc_hiddens, (last_hidden, last_cell) = self.lstm(word_embedding) # sentence_L x batch_size x word_embed_size * 2
        word_contex = self.project(enc_hiddens)

        return word_contex