from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('..')
sys.path.append('../..')

import os.path
import h5py
import numpy as np 
from PIL import Image
import torch
import torch.utils.data as data

from datasets.augmentation import BaseTransform
from utils.config import cfg


__all__ = ['FlowerTextDataset', 'FlowerRawTextDataset']


class FlowerTextDataset(data.Dataset): 
    def __init__(self, image_dir: str, embedding_dir: str, avail_class_dir: str, transform, device='cuda:0', **kwargs):

        self.avail_classes = []
        with open(avail_class_dir, 'r') as avcls:
            while True:
                line = avcls.readline()
                if not line:
                    break
                self.avail_classes.append(line.strip())
        self.avail_classes = sorted(self.avail_classes)
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.device = device
        self.embeddings, self.img_fns, self.idx2clas = self.load_embedding()
        self.transform = transform


    def __len__(self):
        return len(self.embeddings)


    def __getitem__(self, index):
        text_embedding = self.embeddings[index]
        clas = self.idx2clas[index // 10]
        img = self.read_image(os.path.join(self.image_dir, clas, self.img_fns[index // 10]))
        return img, text_embedding


    def load_embedding(self):
        embeddings = []
        all_img_fns = []
        idx2clas = {}

        index = 0
        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0].split('_')[-1])

            txt_fns = os.listdir(os.path.join(self.embedding_dir, clas))
            txt_fns = list(filter(lambda s: os.path.splitext(s)[1] == '.h5' and os.path.splitext(s)[0].isdigit(), txt_fns))

            for i, txt_fn in enumerate(txt_fns):
                txtvals = h5py.File(os.path.join(self.embedding_dir, clas, txt_fn)).values()

                for j, txt in enumerate(txtvals):
                    embeddings.append(np.array(txt, dtype=np.float32))
                idx2clas[index] = clas
                index += 1

            img_fns = os.listdir(os.path.join(self.image_dir, clas))
            img_fns = sorted(img_fns)
            all_img_fns.extend(img_fns)

        return embeddings, all_img_fns, idx2clas


    def read_image(self, img_path):
        image = Image.open(img_path)
        image = np.array(image)
        image = self.transform(image)
        image = image.transpose((2, 0, 1))
        return image

    
    def name(self):
        return "FlowerDataset"



class FlowerRawTextDataset(data.Dataset): 
    def __init__(self, image_dir: str, embedding_dir: str, avail_class_dir: str, transform, device='cuda:0', **kwargs):

        self.avail_classes = []
        with open(avail_class_dir, 'r') as avcls:
            while True:
                line = avcls.readline()
                if not line:
                    break
                self.avail_classes.append(line.strip())
        self.avail_classes = sorted(self.avail_classes)
        self.image_dir = image_dir
        self.embedding_dir = embedding_dir
        self.device = device
        self.embeddings, self.img_fns, self.idx2clas = self.load_embedding()
        self.transform = transform


    def __len__(self):
        return len(self.embeddings)


    def __getitem__(self, index):
        text_embedding = self.embeddings[index]
        clas = self.idx2clas[index // 10]
        img = self.read_image(os.path.join(self.image_dir, clas, self.img_fns[index // 10]))
        return img, text_embedding


    def load_embedding(self):
        embeddings = []
        all_img_fns = []
        idx2clas = {}

        index = 0
        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0].split('_')[-1])

            txt_fns = os.listdir(os.path.join(self.embedding_dir, clas))
            txt_fns = list(filter(lambda s: os.path.splitext(s)[1] == '.h5' and os.path.splitext(s)[0].isdigit(), txt_fns))

            for i, txt_fn in enumerate(txt_fns):
                txtvals = h5py.File(os.path.join(self.embedding_dir, clas, txt_fn)).values()

                for j, txt in enumerate(txtvals):
                    embeddings.append(np.array(txt, dtype=np.float32))
                idx2clas[index] = clas
                index += 1

            img_fns = os.listdir(os.path.join(self.image_dir, clas))
            img_fns = sorted(img_fns)
            all_img_fns.extend(img_fns)

        return embeddings, all_img_fns, idx2clas


    def read_image(self, img_path):
        image = Image.open(img_path)
        image = np.array(image)
        image = self.transform(image)
        image = image.transpose((2, 0, 1))
        return image

    
    def name(self):
        return "FlowerDataset"



if __name__ == '__main__':
    dataset = FlowerTextDataset(image_dir='../../data/flower/images', embedding_dir='../../data/cvpr2016_flowers/text_emb1024',
    	avail_class_dir='../../data/cvpr2016_flowers/allclasses.txt', 
    	transform=BaseTransform(cfg.DATASET.FLOWER.IMAGE_SIZE, cfg.AUG.IMAGE_MEAN, cfg.AUG.IMAGE_STD))
    # print(len(dataset.embeddings))
    # print(len(dataset.img_fns))
    # print(dataset.embeddings[0].shape)
    # print(dataset.img_fns[0])
    for img, text_embedding in dataset:
        print(img.shape)
        print(text_embedding.shape)