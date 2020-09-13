#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch
import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN, WordCNN, WordLSTM
from highway import Highway, SelectiveConnect


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h

        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.e_char = 50
        padding_idx = self.vocab.char_pad
        self.char_embedding = nn.Embedding(len(self.vocab.char2id), self.e_char, padding_idx=padding_idx)
        self.cnn = CNN(embed_size=self.e_char, num_filter=self.word_embed_size)
        self.highway = Highway(embedd_size=self.word_embed_size)
        self.dropout = nn.Dropout(p=0.3)

        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h

        X_words_emb = []
        for X_padded in input:
            X_emb = self.char_embedding(X_padded) # batch_size x max_word_length x char_embed_size
            X_reshaped = torch.transpose(X_emb, dim0=1, dim1=2)

            X_conv_out = self.cnn(X_reshaped)
            X_highway = self.highway(X_conv_out)
            X_word_emb = self.dropout(X_highway)
            X_words_emb.append(X_word_emb)
        X_words_emb = torch.stack(X_words_emb)
        return X_words_emb

        ### END YOUR CODE

class ModelEmbeddings_2(nn.Module):
    def __init__(self, word_embed_size, vocab):
        super(ModelEmbeddings_2, self).__init__()

        self.word_embed_size = word_embed_size
        self.vocab = vocab
        self.e_char = 50
        padding_idx = self.vocab.char_pad
        self.char_embedding = nn.Embedding(len(self.vocab.char2id), self.e_char, padding_idx=padding_idx)
        self.cnn = CNN(embed_size=self.e_char, num_filter=self.word_embed_size)
        self.highway = Highway(embedd_size=self.word_embed_size)

    def forward(self, input):
        X_words_emb = []
        for X_padded in input:
            X_emb = self.char_embedding(X_padded) # batch_size x max_word_length x char_embed_size
            X_reshaped = torch.transpose(X_emb, dim0=1, dim1=2)

            X_conv_out = self.cnn(X_reshaped)
            X_highway = self.highway(X_conv_out)
            X_words_emb.append(X_highway)
        X_words_emb = torch.stack(X_words_emb)
        return X_words_emb


class ContexAwareEmbeddings(nn.Module):

    def __init__(self, word_embed_size, vocab):

        super(ContexAwareEmbeddings, self).__init__()

        self.word_embed_size = word_embed_size
        self.word_embedding = ModelEmbeddings_2(word_embed_size, vocab)
        self.contex_cnn = WordCNN(word_embed_size=word_embed_size, num_filter=word_embed_size)
        self.connect = SelectiveConnect(word_embed_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):

        X_words_emb = self.word_embedding(input) # sentence_L x batch_size x word_embed_size
        X_words_emb_reshaped = X_words_emb.permute(1, 2, 0) # batch_size x word_embed_size x sentence_L
        X_contex = self.contex_cnn(X_words_emb_reshaped) # batch_size x word_embed_size x sentence_L
        X_contex = X_contex.permute(2, 0, 1) # sentence_L x batch_size x word_embed_size
        X_contex_embedding = self.connect(X_contex, X_words_emb)
        #X_contex_embedding = self.dropout(X_contex_embedding)
        return X_contex_embedding


class ContexAwareEmbeddings_LSTM(nn.Module):

    def __init__(self, word_embed_size, vocab):

        super(ContexAwareEmbeddings_LSTM, self).__init__()
        self.word_embed_size = word_embed_size
        self.word_embedding = ModelEmbeddings_2(word_embed_size, vocab)
        self.contex_lstm = WordLSTM(word_embed_size=word_embed_size)
        self.connect = SelectiveConnect(word_embed_size)
        self.dropout = nn.Dropout(p=0.3)


    def forward(self, input):

        X_words_emb = self.word_embedding(input)
        X_contex_lstm = self.contex_lstm(X_words_emb)
        X_contex_embedding = self.connect(X_contex_lstm, X_words_emb)
        #X_contex_embedding = self.dropout(X_contex_embedding)
        return X_contex_embedding