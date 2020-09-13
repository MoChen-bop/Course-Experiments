import sys
sys.path.append('../..')

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F 
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.config import cfg 


class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5, nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.MODEL.MIRRORGAN.TEXT.WORDS_NUM
        self.ntoken = ntoken
        self.ninput = ninput
        self.drop_prob = drop_prob
        self.nlayers = nlayers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.MODEL.MIRRORGAN.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()


    def define_module():
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.ninput, self.nhidden, self.nlayers, batch_first=True,
            	dropout=self.drop_prob, bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden, self.nlayers, batch_first=True,
            	dropout=self.drop_prob, bidirectional=self.bidirectional)
        else:
            raise NotImplementedError


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
            	Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directiona, bsz, self.nhidden).zero_())


    def forward(self, captions, cap_lens, hidden, mask=None):
    	# captions: batch_size x n_steps
        emb = self.drop(self.encoder(captions))
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)

        output, hidden = self.rnn(emb, hidden)
        output = pad_packed_sequence(output, batch_first=True)[0]

        words_emb = output.transpose(1, 2) # batch_size x hidden_size * num_directions x seq_len

        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
    def __init__(self, nef):
        super(CNN_ENCODER, self).__init__()
        if cfg.MODEL.MIRRORGAN.TRAIN_FLAG:
            self.nef = nef
        else:
            self.nef = 256

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_goole-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print("Load pretrained model from ", url)

        self.define_module(model)
        self.init_trainable_weights()


    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.nef)
        self.emb_cnn_code = nn.Linear(2048, self.nef)


    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


    def forward(self, x):
        features = None

        x = nn.Upsample(size=(299, 299), mode='bilinear')(x) # batch_size x 3 x 299 x 299

        x = self.Conv2d_1a_3x3(x) # batch_size x 32 x 149 x 149
        x = self.Conv2d_1a_3x3(x) # batch_size x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x) # batch_size x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2) # batch_size x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x) # batch_size x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x) # batch_size x 192 x 71 x 71

        x = F.max_pool2d(x, kernel_size=3, stride=2) # batch_size x 192 x 35 x 35
        x = self.Mixed_5b(x) # batch_size x 256 x 35 x 35
        x = self.Mixed_5c(x) # batch_size x 288 x 35 x 35
        x = self.Mixed_5d(x) # batch_size x 288 x 35 x 35

        x = self.Mixed_6a(x) # batch_size x 768 x 17 x 17
        x = self.Mixed_6b(x) # batch_size x 768 x 17 x 17
        x = self.Mixed_6c(x) # batch_size x 768 x 17 x 17
        x = self.Mixed_6d(x) # batch_size x 768 x 17 x 17
        x = self.Mixed_6e(x) # batch_size x 768 x 17 x 17

        features = x # batch_size x 768 x 17 x 17

        x = self.Mixed_7a(x) # batch_size x 1280 x 8 x 8
        x = self.Mixed_7b(x) # batch_size x 2048 x 8 x 8
        x = self.Mixed_7c(x) # batch_size x 2048 x 8 x 8
        x = F.avg_pool2d(x, kernel_size=8) # batch_size x 2048 x 1 x 1
        x = x.view(x.size(0), -1) # batch_size x 2048

        cnn_code = self.emb_cnn_code(x) # batch_size x nef
        if features is not None:
            features = self.emb_features(features) # batch_size x nef x 17 x 17
        return features, cnn_code