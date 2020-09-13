from __future__ import print_function

import argparse
import logging
import os
import sys
sys.path.append('..')
import six
import time
import random
import numpy as np

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from models.deepfm import DeepFM
from models.dnn_2 import DNN
from models.drnn_2 import DRNN
from models.dnn_plus import DNNPlus
from utils.config import config as cfg
from utils.config import print_config, update_config
from utils.option import BaseOptions
from datasets.criteo_dataset_reader import CriteoDataset, data_reader
from train_dygraph_2 import train

if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    # exp1
    cfg.train_model = 'deepfm'
    train()
    cfg.train_model = 'dnn'
    train()
    cfg.train_model = 'dnnplus'
    train()

    # exp2
    cfg.train_model = 'drnn'
    cfg.drnn_hidden_layer = 1
    cfg.drnn_hidden_dim = 128
    train()
    cfg.drnn_hidden_layer = 2
    cfg.drnn_hidden_dim = 128
    train()
    cfg.drnn_hidden_layer = 4
    cfg.drnn_hidden_dim = 128
    train()
    cfg.drnn_hidden_layer = 1
    cfg.drnn_hidden_dim = 256
    train()
    cfg.drnn_hidden_layer = 2
    cfg.drnn_hidden_dim = 256
    train()
    cfg.drnn_hidden_layer = 4
    cfg.drnn_hidden_dim = 256
    train()
    cfg.drnn_hidden_layer = 2
    cfg.drnn_hidden_dim = 512
    train()
