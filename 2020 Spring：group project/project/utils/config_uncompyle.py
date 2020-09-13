# uncompyle6 version 3.7.2
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.16 |Anaconda, Inc.| (default, Mar 14 2019, 21:00:58) 
# [GCC 7.3.0]
# Embedded file name: ../utils/config.py
# Compiled at: 2020-06-17 11:03:12
from __future__ import print_function
from __future__ import unicode_literals
from utils.collect import Config
import numpy as np
cfg = Config()
cfg.DEVICE = 'cuda:0'
cfg.BATCH_SIZE = 4
cfg.AUG.IMAGE_MEAN = [
 0.485, 0.456, 0.406]
cfg.AUG.IMAGE_STD = [0.229, 0.224, 0.225]
cfg.DATASET.DATA_FEEDER_NUM = 2
cfg.DATASET.TEXT_EMBEDDING_CUB.DATASET_DIR = '../data/cvpr2016_cub'
cfg.DATASET.TEXT_EMBEDDING_CUB.AVAIL_CLASS_FN = 'allclasses.txt'
cfg.DATASET.TEXT_EMBEDDING_CUB.IMAGE_DIR = 'images_h5'
cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_DIR = 'text_c10'
cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_CUTOFF = 200
cfg.DATASET.TEXT_EMBEDDING_FLOWER.DATASET_DIR = '../data/cvpr2016_flowers'
cfg.DATASET.TEXT_EMBEDDING_FLOWER.AVAIL_CLASS_FN = 'allclasses.txt'
cfg.DATASET.TEXT_EMBEDDING_FLOWER.IMAGE_DIR = 'images_h5'
cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_DIR = 'text_c10'
cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_CUTOFF = 200
cfg.DATASET.COCO.IMAGE_SIZE = 512
cfg.DATASET.COCO.CATEGORIES = 90
cfg.DATASET.COCO.TRAIN_IMAGE_DIR = '/mnt/2020project/Assets/Datasets/coco/images/train2017'
cfg.DATASET.COCO.TRAIN_IMAGE_LABEL_DIR = '/mnt/2020project/Assets/Datasets/coco/annotations/image_label_train2017.json'
cfg.DATASET.COCO.TRAIN_INSTANCES_DIR = '/mnt/2020project/Assets/Datasets/coco/annotations/instances_train2017.json'
cfg.TRAIN.SAVE_DIR = '../saved_models'
cfg.TRAIN.LOG_DIR = '../logs'
cfg.TRAIN.LEARNING_RATE = 0.0001
cfg.TRAIN.TEXT_EMBEDDING.TEXT_EMBEDDING_DATASET = 'cub'
cfg.TRAIN.TEXT_EMBEDDING.LR_DECAY = True
cfg.TRAIN.TEXT_EMBEDDING.MAX_STEP = 400000
cfg.TRAIN.TEXT_EMBEDDING.LOG_INTERVAL = 100
cfg.TRAIN.TEXT_EMBEDDING.SAVE_INTERVAL = 10000
cfg.TRAIN.TEXT_EMBEDDING.RESUME_DIR = '../saved_models/CRNN/CUBDataset/CRNN_69_348_512_512_4_4_4_3_3_3_LSTM_512_1_180000.pth'
cfg.TRAIN.XCEPTION.RESUME_DIR = ''
cfg.TRAIN.XCEPTION.MAX_STEP = 100
cfg.TRAIN.XCEPTION.LOG_INTERVAL = 32
cfg.TRAIN.XCEPTION.SAVE_INTERVAL = 1
cfg.TRAIN.XCEPTION.RESUME_DIR = '../saved_models/Xception/COCOClsDataset/Xception_0.pth'
cfg.MODEL.CRNN.CONV_CHANNELS = [
 348, 512, 512]
cfg.MODEL.CRNN.CONV_KERNELS = [4, 4, 4]
cfg.MODEL.CRNN.CONV_STRIDES = [3, 3, 3]
cfg.MODEL.CRNN.RNN_BIDIR = False
cfg.MODEL.CRNN.RNN_DROPOUT = 0.3
cfg.MODEL.CRNN.LIN_DROPOUT = 0
cfg.MODEL.CRNN.CONV_DROPOUT = 0
cfg.MODEL.CRNN.RNN_HIDDEN_SIZE = 512
cfg.MODEL.CRNN.RNN_NUM_LAYERS = 1
cfg.MODEL.CRNN.LSTM = True
# okay decompiling ./config.pyc
