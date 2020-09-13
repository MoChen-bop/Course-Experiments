from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('..')
sys.path.append('../..')
import random
import pickle

import os.path
import h5py
import numpy as np 
from PIL import Image
import torch
import torch.utils.data as data

from datasets.augmentation import BaseTransform
from utils.config import cfg


__all__ = ['CUBTextDataset']


class CUBTextDataset(data.Dataset): 
    def __init__(self, image_dir: str, embedding_dir: str, transform, device='cuda:0', **kwargs):

        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.device = device
        self.embeddings, self.filenames = self.load_embedding()
        self.transform = transform


    def __len__(self):
        return len(self.embeddings)


    def __getitem__(self, index):
        rand_idx = random.randint(0, 9)
        text_embedding = self.embeddings[index][rand_idx]
        image_name = self.filenames[index] + '.jpg'
        img_path = os.path.join(self.image_dir, image_name)
        img = self.read_image(img_path)
        return img, text_embedding


    def load_embedding(self):
        with open(os.path.join(self.embedding_dir, 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
            embeddings = pickle.load(f, encoding='latin')

        with open(os.path.join(self.embedding_dir, 'filenames.pickle'), 'rb') as f:
            filenames = pickle.load(f, encoding='latin')
        return embeddings, filenames


    def read_image(self, img_path):
        image = Image.open(img_path)
        image = np.array(image)
        image = self.transform(image)
        image = image.transpose((2, 0, 1))
        return image


    def name(self):
        return "CUBDataset"



if __name__ == '__main__':
    dataset = CUBTextDataset(image_dir='../../data/bird/CUB_200_2011/CUB_200_2011/images', embedding_dir='../../data/emb_hdgan_bird/all',
    	transform=BaseTransform(cfg.DATASET.CUB.IMAGE_SIZE, cfg.AUG.IMAGE_MEAN, cfg.AUG.IMAGE_STD))
    # print(len(dataset.embeddings))
    # print(len(dataset.img_fns))
    # print(dataset.embeddings[0].shape)
    # print(dataset.img_fns[0])
    for img, text_embedding in dataset:
        print(img.shape)
        print(text_embedding.shape)