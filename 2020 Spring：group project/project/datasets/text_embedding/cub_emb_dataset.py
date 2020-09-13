import numpy as np 
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import pickle
import random

import torch
from torch.utils import data


class CUBTextEmbeddingDataset(data.Dataset):
    
    def __init__(self, dataset_dir: str, captions_dir: str, text_cutoff: int):
        super().__init__()

        with open(os.path.join(dataset_dir, 'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
            self.caption_embeddings = pickle.load(f, encoding='latin')

        with open(os.path.join(dataset_dir, 'filenames.pickle'), 'rb') as f:
            self.filenames = pickle.load(f, encoding='latin')

        with open(os.path.join(dataset_dir, 'caption_path.pickle'), 'rb') as f:
            self.caption_path_dict = pickle.load(f, encoding='latin')

        self.captions_dir = captions_dir
        self.captions = self.load_captions()

        alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
        self.vocab = {k: i for i, k in enumerate(alphabet)}
        self.vocab['ï'] = self.vocab['i']
        self.vocab['¿'] = self.vocab['?']
        self.vocab['½'] = self.vocab[' ']
        self.vocab['�'] = self.vocab['.']
        self.vocab_len = len(self.vocab) - 4

        self.text_cutoff = text_cutoff


    def __len__(self):
        return len(self.filenames)


    def __getitem__(self, index):
        rand_idx = random.randint(0, 9)
        embedding = self.caption_embeddings[index][rand_idx]
        caption = self.captions[index]
        caption_onehot = self.process_text(caption[rand_idx])
        caption_onehot = torch.from_numpy(caption_onehot.astype(np.float32))
        embedding = torch.from_numpy(embedding.astype(np.float32))

        return caption_onehot, embedding


    def load_captions(self):
        captions = []
        for fn in self.filenames:
            fn = fn.split('/')[-1]
            caption_path = os.path.join(self.captions_dir, self.caption_path_dict[fn])
            caps = []
            with open(caption_path, 'r') as f:
                for line in f.readlines():
                    caps.append(line.strip())
            captions.append(caps)
        return captions


    def process_text(self, text):
        text = text[:self.text_cutoff]
        onehot = np.zeros((self.vocab_len, self.text_cutoff))
        onehot[tuple([[self.vocab[tok] for tok in text], range(len(text))])] = 1
        return onehot


    def name(self):
        return 'CUBDataset'


if __name__ == '__main__':
    dataset = CUBTextEmbeddingDataset('../../data/emb_bird/all', '../../data/emb_bird', 200)
    print(len(dataset))
    # print(dataset[0][0])
    # print(dataset[0][1])
    for caption, embedding in dataset:
        print(caption.shape)
        print(embedding.shape)
        sys.stdout.flush()