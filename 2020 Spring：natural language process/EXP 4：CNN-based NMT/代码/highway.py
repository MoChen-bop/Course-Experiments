#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # pass
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self, embedd_size):
        super(Highway, self).__init__()
        self.relu = nn.ReLU()
        self.projection = nn.Linear(embedd_size, embedd_size)
        self.gate = nn.Linear(embedd_size, embedd_size)


    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        X_proj = self.projection(X_conv_out)
        X_proj = self.relu(X_proj)
        X_gate = torch.sigmoid(self.gate(X_conv_out))
        X_highway = torch.mul(X_proj, X_gate) + torch.mul(X_conv_out, 1 - X_gate)

        return X_highway

    ### END YOUR CODE


class SkipConnect(nn.Module):
    
    def __init__(self, embedd_size):
        super(SkipConnect, self).__init__()
        self.relu = nn.ReLU()
        self.projection = nn.Linear(embedd_size, embedd_size)
        

    def forward(self, X_conv_out: torch.Tensor) -> torch.Tensor:
        X_proj = self.projection(X_conv_out)
        X_skip_connect = X_proj + X_conv_out
        X_skip_connect = self.relu(X_skip_connect)

        return X_skip_connect


class SelectiveConnect(nn.Module):
    
    def __init__(self, word_embed_size):
        super(SelectiveConnect, self).__init__()
        self.gate = nn.Linear(word_embed_size + word_embed_size, word_embed_size)
        #self.gate = nn.Linear(word_embed_size, word_embed_size)


    def forward(self, X_contex, X_embedding):
        X_concat = torch.cat([X_contex, X_embedding], dim=2)
        X_gate = torch.sigmoid(self.gate(X_concat))
        #X_gate = torch.sigmoid(self.gate(X_contex))
        X_contex_embedding = torch.mul(X_contex, 1 - X_gate) + torch.mul(X_embedding, X_gate)
        return X_contex_embedding
        # return X_contex
