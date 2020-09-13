import os
import sys
sys.path.append('..')
import pickle
import numpy as np
import json
import random
import cv2

import torch
from torch.utils.data import Dataset

from utils.config import cfg
from datasets.augmentation import Augmentation, BaseTransform



def cv2_imread(file_path, flag=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flag)


def save_image(image, image_dir):
    image = np.array(image).astype(np.float32)
    image = image.transpose((1, 2, 0))
    image *= cfg.AUG.IMAGE_STD
    image += cfg.AUG.IMAGE_MEAN
    image *= 255.0
    image = image.astype(np.uint8)

    cv2.imwrite(image_dir, image)

    


class COCOClsDataset(Dataset):
    
    def __init__(self, image_dir, image_label_dir, instances_dir, transform=None):
        self.image_dir = image_dir
        self.image_label_list = self._get_image_list(image_label_dir)
        self.label_names = self._get_label_names(instances_dir)
        if transform is None:
            self.transform = BaseTransform(size=cfg.DATASET.COCO.IMAGE_SIZE,
                mean=cfg.AUG.IMAGE_MEAN, std=cfg.AUG.IMAGE_STD)
        else:
            self.transform = transform
        self.categories = cfg.DATASET.COCO.CATEGORIES


    def __len__(self):
        return len(self.image_label_list)


    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_label_list[index]['image_path'])
        label = int(self.image_label_list[index]['label'] - 1)
        image = self._load_image(image_path)
        image = self.transform(image)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        label = torch.from_numpy(np.array(label))

        return image, label



    def _get_image_list(self, image_label_dir):
        f = open(image_label_dir)
        image_label_dict = json.load(f)
        image_label_list = []
        for k, v in image_label_dict.items():
            image_label_list.append(v)
        f.close()
        return image_label_list


    def _get_label_names(self, instances_dir):
        f = open(instances_dir)
        instances_dict = json.load(f)
        label_names = {}
        for cate in instances_dict['categories']:
            label_names[int(cate['id'])] = cate['name']
        f.close()
        return label_names


    def _load_image(self, img_path):
        cv2_imread_flag = cv2.IMREAD_COLOR
        img = cv2_imread(img_path, cv2_imread_flag)

        return img
    

    def name(self):
        return 'COCOClsDataset'


if __name__ == '__main__':
    transform = Augmentation(size=cfg.DATASET.COCO.IMAGE_SIZE,
        mean=cfg.AUG.IMAGE_MEAN, std=cfg.AUG.IMAGE_STD)
    dataset = COCOClsDataset('../data/coco2014/train2014', 
    	'../data/coco2014/annotations/image_label_train2014.json',
    	'../data/coco2014/annotations/instances_train2014.json',
    	transform)
    print(len(dataset))
    if not os.path.exists('../visualize/augmentation/coco'):
        os.makedirs('../visualize/augmentation/coco')

    for i in range(20):
        image, _ = dataset[i]
        save_image(image, '../visualize/augmentation/coco/' + str(i) + '.jpg')