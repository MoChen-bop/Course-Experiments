import numpy as np 
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import pickle
import random

import torch
from torch.utils import data


class COCOTextEmbeddingDataset(data.Dataset):
    
    def __init__(self, dataset_dir: str, captions_dir: str, text_cutoff: int):
        super().__init__()

        with open(os.path.join(dataset_dir, 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
            self.caption_embeddings = pickle.load(f, encoding='latin')

        with open(os.path.join(dataset_dir, 'filenames.pickle'), 'rb') as f:
            self.filenames = pickle.load(f, encoding='latin')

        with open(os.path.join(dataset_dir, 'fn2caption.pickle'), 'rb') as f:
            self.caption_dict = pickle.load(f, encoding='latin')

        self.captions = self.load_captions()

        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        self.vocab = {k: i for i, k in enumerate(alphabet)}
        self.vocab['ï'] = self.vocab['i']
        self.vocab['¿'] = self.vocab['?']
        self.vocab['½'] = self.vocab[' ']
        self.vocab_len = len(self.vocab) - 3

        self.text_cutoff = text_cutoff


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        rand_idx = random.randint(0, 4)
        embedding = self.caption_embeddings[index][rand_idx]
        caption = self.captions[index]
        caption_onehot = self.process_text(caption[rand_idx].lower().replace('\n', ' '))
        caption_onehot = torch.from_numpy(caption_onehot.astype(np.float32))
        embedding = torch.from_numpy(embedding.astype(np.float32))

        return caption_onehot, embedding


    def load_captions(self):
        captions = []
        for fn in self.filenames:
            fn = str(fn)[2:-5]
            captions.append(self.caption_dict[fn])
        return captions


    def process_text(self, text):
        text = text[:self.text_cutoff]
        onehot = np.zeros((self.vocab_len, self.text_cutoff))
        onehot[tuple([[self.vocab[tok] for tok in text], range(len(text))])] = 1
        return onehot


    def name(self):
        return 'COCODataset'


if __name__ == '__main__':
    dataset = COCOTextEmbeddingDataset('../../data/emb_coco/coco/coco/train', 'dummy', 200)
    print(len(dataset))
    # print(dataset[0][0])
    # print(dataset[0][1])
    i = 0
    for caption, embedding in dataset:
        print(caption.shape)
        print(embedding.shape)
        i += 1
        if i > 100:
            break
        sys.stdout.flush()