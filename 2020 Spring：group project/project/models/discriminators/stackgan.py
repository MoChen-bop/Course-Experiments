import os
import sys
sys.path.append('..')
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable

from utils.config import cfg


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    	             padding=1, bias=False)


class D_GET_LOGITS(nn.Module):
    
    def __init__(self, ndf, nef, bcondition=True):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if bcondition:
            self.outlogits = nn.Sequential(
            	conv3x3(ndf * 8 + nef, ndf * 8),
            	nn.BatchNorm2d(ndf * 8),
            	nn.LeakyReLU(0.2, inplace=True),
            	nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            	nn.Sigmoid())
        else:
            self.outlogits = nn.Sequential(
            	nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            	nn.Sigmoid())


    def forward(self, h_code, c_code=None):
        if self.bcondition and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)

            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code) # batch_size x 1
        return output.view(-1)


class STAGE1_D(nn.Module):

    def __init__(self, df_dim, condition_dim):
        super(STAGE1_D, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = condition_dim
        self.define_module()


    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
        	nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf x 32 x 32

        	nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 2),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 2 x 16 x 16

        	nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 4),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 4 x 8 x 8

        	nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 8), # batch_size x ndf * 8 x 4 x 4

        	nn.LeakyReLU(0.2, inplace=True))

        self.get_cond_logits = D_GET_LOGITS(ndf, nef)
        self.get_uncond_logits = None


    def forward(self, image):
        img_embedding = self.encode_img(image) # batch_size x (ndf * 8) x 4 x 4

        return img_embedding


    def name(self):
        return "Stage1_D"


class STAGE2_D(nn.Module):

    def __init__(self, df_dim, condition_dim):
        super(STAGE2_D, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = condition_dim
        self.define_module()


    def define_module(self):
        ndf, nef = self.df_dim, self.ef_dim
        self.encode_img = nn.Sequential(
        	nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf x 128 x 128

        	nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 2),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 2 x 64 x 64

        	nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 4),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 4 x 32 x 32

        	nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 8),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 8 x 16 x 16

        	nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 16),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 16 x 8 x 8

        	nn.Conv2d(ndf * 16, ndf * 32, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ndf * 32),
        	nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 32 x 4 x 4

        	conv3x3(ndf * 32, ndf * 16),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True), # batch_size x ndf * 16 x 4 x 4

        	conv3x3(ndf * 16, ndf * 8),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)) # batch_size x ndf * 8 x 4 x 4

        self.get_cond_logits = D_GET_LOGITS(ndf, nef, bcondition=True)
        self.get_uncond_logits = D_GET_LOGITS(ndf, nef, bcondition=False)


    def forward(self, image):
        img_embedding = self.encode_img(image)

        return img_embedding


    def name(self):
        return "Stage2_D"
