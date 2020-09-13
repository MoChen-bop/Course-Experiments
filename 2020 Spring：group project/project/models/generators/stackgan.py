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


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
    	nn.Upsample(scale_factor=2, mode='nearest'),
    	conv3x3(in_planes, out_planes),
    	nn.BatchNorm2d(out_planes),
    	nn.ReLU(True))
    return block


class ResBlock(nn.Module):
    
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
        	conv3x3(channel_num, channel_num),
        	nn.BatchNorm2d(channel_num),
        	nn.ReLU(True),
        	conv3x3(channel_num, channel_num),
        	nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(x)
        return out


class CA_NET(nn.Module):
    
    def __init__(self, t_dim, c_dim):
        super(CA_NET, self).__init__()
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()
    

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar
    

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(cfg.DEVICE)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class STAGE1_G(nn.Module):
    
    def __init__(self, t_dim, gf_dim, condition_dim, z_dim):
        super(STAGE1_G, self).__init__()
        self.text_dim = t_dim
        self.gf_dim = gf_dim * 8
        self.ef_dim = condition_dim
        self.z_dim = z_dim
        self.define_module()


    def define_module(self):
        ninput = self.z_dim + self.ef_dim
        ngf = self.gf_dim

        self.ca_net = CA_NET(self.text_dim, self.ef_dim)

        self.fc = nn.Sequential(
        	nn.Linear(ninput, ngf * 4 * 4, bias=False),
        	nn.BatchNorm1d(ngf * 4 * 4),
        	nn.ReLU(True))

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

        self.img = nn.Sequential(
        	conv3x3(ngf // 16, 3),
        	nn.Tanh())

    
    def forward(self, text_embedding, noise):
        c_code, mu, logvar = self.ca_net(text_embedding) # batch_size x condition_dim
        z_c_code = torch.cat((noise, c_code), 1) # batch_size x (noise_dim + gf_dim * 4 * 4)
        h_code = self.fc(z_c_code) # batch_size x (gf_dim * 4 * 4)

        h_code = h_code.view(-1, self.gf_dim, 4, 4) # batch_size x gf_dim x 4 x 4
        h_code = self.upsample1(h_code) # batch_size x ngf // 2 x 8 x 8
        h_code = self.upsample2(h_code) # batch_size x ngf // 4 x 16 x 16
        h_code = self.upsample3(h_code) # batch_size x ngf // 8 x 32 x 32
        h_code = self.upsample4(h_code) # batch_size x ngf // 16 x 64 x 64

        fake_img = self.img(h_code) # batch_size x 3 x 64 x 64

        return None, fake_img, mu, logvar
    

    def name(self):
        return "Stage1_G"


class STAGE2_G(nn.Module):

    def __init__(self, STAGE1_G, gf_dim, condition_dim, z_dim, deploy=False, stage2=True):
        super(STAGE2_G, self).__init__()
        self.gf_dim = gf_dim
        self.ef_dim = condition_dim
        self.z_dim = z_dim
        self.STAGE1_G = STAGE1_G

        for param in self.STAGE1_G.parameters():
            param.requires_grad = False
        self.define_module()

        self.deploy = deploy
        self.stage2 = stage2


    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.MODEL.STACKGAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)


    def define_module(self):
        ngf = self.gf_dim
        self.ca_net = CA_NET(self.STAGE1_G.text_dim, self.ef_dim)
        self.encoder = nn.Sequential(
        	conv3x3(3, ngf),
        	nn.ReLU(True), # batch_size x ngf x 64 x 64

        	nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ngf * 2),
        	nn.ReLU(True), # batch_size x ngf * 2 x 32 x 32

        	nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
        	nn.BatchNorm2d(ngf * 4),
        	nn.ReLU(True)) # batch_size x ngf * 4 x 16 x 16
        self.hr_joint = nn.Sequential(
        	conv3x3(self.ef_dim + ngf * 4, ngf * 4),
        	nn.BatchNorm2d(ngf * 4),
        	nn.ReLU(True)) # batch_size x ngf * 4 x 16 x 16
        self.residual = self._make_layer(ResBlock, ngf * 4)

        self.upsample1 = upBlock(ngf * 4, ngf * 2) # batch_size x ngf * 2 x 32 x 32
        self.upsample2 = upBlock(ngf * 2, ngf) # batch_size x ngf x 64 x 64
        self.upsample3 = upBlock(ngf, ngf // 2) # batch_size x ngf // 2 x 128 x 128
        self.upsample4 = upBlock(ngf // 2, ngf // 4) # batch_size x ngf // 4 x 256 x 256
        self.img = nn.Sequential(
        	conv3x3(ngf // 4, 3),
        	nn.Tanh()) # batch_size x 3 x 256 x 256


    def forward(self, text_embedding, noise):
        _, stage1_img, _, _ = self.STAGE1_G(text_embedding, noise) # batch_size x 3 x 64 x 64
        stage1_img = stage1_img.detach()
        encoded_img = self.encoder(stage1_img) # batch_size x ndf x 8 x 8

        c_code, mu, logvar = self.ca_net(text_embedding)
        c_code = c_code.view(-1, self.ef_dim, 1, 1) # batch_size x ef_dim x 1 x 1
        c_code = c_code.repeat(1, 1, 16, 16) # batch_size x ef_dim x 16 x 16
        i_c_code = torch.cat([encoded_img, c_code], 1)
        h_code = self.hr_joint(i_c_code) # batch_size x ngf * 4 x 16 x 16
        h_code = self.residual(h_code) # batch_size x ngf * 4 x 16 x 16

        h_code = self.upsample1(h_code) # batch_size x ngf * 2 x 32 x 32
        h_code = self.upsample2(h_code) # batch_size x ngf x 64 x 64
        h_code = self.upsample3(h_code) # batch_size x ngf // 2 x 128 x 128
        h_code = self.upsample4(h_code) # batch_size x ngf // 4 x 256 x 256

        fake_img = self.img(h_code) # batch_size x 3 x 256 x 256

        if self.deploy:
            if self.stage2:
                return fake_img
            else:
                return stage1_img
        else:
            return stage1_img, fake_img, mu, logvar


    def name(self):
        return "Stage2_G"