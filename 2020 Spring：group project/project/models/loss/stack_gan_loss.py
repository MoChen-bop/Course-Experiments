import os
import sys
sys.path.append('../..')
import errno
import numpy as np 

from copy import deepcopy

import torch
import torch.nn as nn
from torch.nn import init
import torchvision.utils as vutils

from utils.config import cfg


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def compute_discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, conditions):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()

    real_features = netD(real_imgs)
    fake_features = netD(fake)

    real_logits = netD.get_cond_logits(real_features, cond)
    errD_real = criterion(real_logits, real_labels)

    wrong_logits = netD.get_cond_logits(real_features[:(batch_size - 1)], cond[1:])
    errD_wrong = criterion(wrong_logits, fake_labels[1:])

    fake_logits = netD.get_cond_logits(fake_features, cond)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = netD.get_uncond_logits(real_features)
        fake_logits = netD.get_uncond_logits(fake_features)

        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)

        errD = ((errD_real + uncond_errD_real) / 2. + 
        	(errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.item(), errD_wrong.item(), errD_fake.item()


def compute_generator_loss(netD, fake_imgs, real_labels, conditions):
    criterion = nn.BCELoss()
    cond = conditions.detach()

    fake_features = netD(fake_imgs)
    fake_logits = netD.get_cond_logits(fake_features, cond)

    errD_fake = criterion(fake_logits, real_labels)

    if netD.get_uncond_logits is not None:
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake