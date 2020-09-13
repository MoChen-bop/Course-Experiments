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
from models.generators.mirrorgan_glattention import GLAttentionGeneral as ATT_NET


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()


    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0 "channels don't divide 2!"
        nc = it(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def upBlock(in_planes, out_planes):
    block = nn.Sequential(
    	nn.Upsample(scale_factor=2, mode='nearest'),
    	conv3x3(in_planes, out_planes * 2),
    	nn.BatchNorm2d(out_planes * 2),
    	GLU())
    return block 


def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
    	conv3x3(in_planes, out_planes * 2),
    	nn.BatchNorm2d(out_planes * 2),
    	GLU())
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
        	conv3x3(channel_num, channel_num * 2),
        	nn.BatchNorm2d(channel_num * 2),
        	GLU(),
        	conv3x3(channel_num, channel_num),
        	nn.BatchNorm2d(channel_num))


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out 


class CA_NET(nn.Module):
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.MODEL.MIRRORGAN.TEXT.EMBEDDING_DIM
        self.c_dim = cfg.MODEL.MIRRORGAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()


    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding)) # batch_size x (c_dim * 2)
        mu = x[:, :self.c_dim] # batch_size x c_dim
        logvar = x[:, self.c_dim:] # batch_size x c_dim
        return mu, logvar


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(cfg.DEVICE)
        eps = Variable(eps)
        return eps.mul(std).add_(mu) # batch_size x c_dim


    def forward(self, text_embedding):
        mu, logvar = self.encoder(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = cfg.MODEL.MIRRORGAN.Z_DIM + ncf

        self.define_module()


    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
        	nn.Linear(nz, ngf * 4 * 4 * 3, bias=False),
        	nn.BatchNorm1d(ngf * 4 * 4 * 2),
        	GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)


    def forward(self, z_code, c_code):
        c_z_code = torch.cat((c_code, z_code), 1) # batch_size x (z_dim + c_dim)
        out_code = self.fc(c_z_code) # batch_size x (ngf * 4 * 4)
        out_code4 = out_code.view(-1, self.gf_dim, 4, 4) # batch_size x ngf x 4 x 4

        out_code8 = self.upsample1(out_code4) # batch_size x ngf // 2 x 8 x 8
        out_code16 = self.upsample2(out_code8) # batch_size x ngf // 4 x 16 x 16
        out_code32 = self.upsample3(out_code16) # batch_size x ngf // 8 x 32 x 32
        out_code64 = self.upsample4(out_code32) # batch_size x ngf // 16 x 64 x 64

        return out_code64


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf

        self.num_residual = cfg.MODEL.MIRRORGAN.R_NUM
        self.define_module()
        self.conv = conv1x1(ngf * 3, ngf * 2)


    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(cfg.MODEL.MIRRORGAN.R_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)


    def define_module(self):
        ngf = self.gf_dim
        self.att = ATT_NET(ngf, self.ef_dim)
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)


    def forward(self, h_code, c_code, word_embs, mask):
        self.att.applyMask(mask)

        c_code, weightedSentence, att, sent_att = self.att(h_code, c_code, word_embs)
        h_c_code = torch.cat((h_code, c_code), 1)
        h_c_sent_code = torch.cat((h_c_code, weightedSentence), 1) # batch_size x (ngf * 3) x ih x iw

        h_c_sent_code = self.conv(h_c_sent_code) # batch_size x (ngf * 2) x ih x iw
        out_code = self.residual(h_c_sent_code) # batch_size x (ngf * 2) x ih x iw

        out_code = self.upsample(out_code) # batch_size x ngf x (ih * 2) x (iw * 2)
        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_fim = ngf
        self.img = nn.Sequential(
        	conv3x3(ngf, 3),
        	nn.Tanh())


    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        ngf = cfg.MODEL.MIRRORGAN.GF_DIM 
        nef = cfg.MODEL.MIRRORGAN.EMBEDDING_DIM
        ncf = cfg.MODEL.MIRRORGAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)
            self.img_net1 = GET_IMAGE_G(ngf)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net2 = GET_IMAGE_G(ngf)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)
            self.img_net3 = GET_IMAGE_G(ngf)


    def forward(self, z_code, sent_emb, word_embs, mask):
        fake_imgs = []
        att_maps = []

        c_code, mu, logvar = self.ca_net(sent_emb) # batch_size x c_dim

        if cfg.MODEL.MIRRORGAN.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)

        if cfg.MODEL.MIRRORGAN.BRANCH_NUM > 1:
            h_code2, att = self.h_net2(h_code1, c_code, word_embs, mask)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)

        if cfg.MODEL.MIRRORGAN.BRANCH_NUM > 2:
            h_code3, att2 = self.h_net3(h_code2, c_code, word_embs, mask)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_imgs)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, mu, logvar


class G_DCGAN(nn.Module):
    def __init__(self):
        super(G_DCGAN, self).__init__()
        ngf = cfg.MODEL.MIRRORGAN.GF_DIM
        nef = cfg.MODEL.MIRRORGAN.TEXT.EMBEDDING_DIM
        ncf = cfg.MODEL.MIRRORGAN.CONDITION_DIM
        self.ca_net = CA_NET()

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf)

        self.img_net = GET_IMAGE_G(ngf)


    def forward(self, z_code, sent_emb, word_embs, mask):
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 0:
            h_code = self.h_net1(z_code, c_code)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 1:
            h_code, att1 = self.h_net2(h_code, c_code, word_embs, mask)
            if att1 is not None:
                att_amps.append(att1)

        if cfg.MODEL.MIRRORGAN.TREE.BRANCH_NUM > 2:
            h_code, att2 = self.h_net3(h_code, c_code, word_embs, mask)
            if att2 is not None:
                att_maps.append(att2)

        fake_imgs = self.img_net(h_code)
        return [fake_imgs], att_maps, mu, logvar