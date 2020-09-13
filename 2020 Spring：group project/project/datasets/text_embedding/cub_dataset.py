import os
import string
import numpy as np
import torch
import torchfile
import h5py
from torch.utils import data

__all__ = ['CUBTextEmbeddingDataset', 'CUBTextEmbeddingDatasetLazy']


class CUBTextEmbeddingDataset(data.Dataset):
    
    def __init__(self, dataset_dir: str, avail_class_fn: str, image_dir: str,
    	         text_dir: str, text_cutoff: int, level='char', device='cuda:0', **kwargs):
        super().__init__()

        assert level in ('word', 'char')

        if level == 'word':
            assert 'vocab_fn' in kwargs

            vocab_fn = kwargs['vocab_fn']
            vocab = torchfile.load(os.path.join(dataset_dir, vocab_fn))

            self.vocab = {k.decode('utf-8'): vocab[k] - 1 for k in vocab}
            self.vocab_len = len(self.vocab)
            self.split = lambda s: list(filter(
            	lambda ss: ss,
            	map(
            		lambda ss: ss.translate(str.maketrans('', '', string.punctuation)),
            		s.split())))
        else:
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
            self.vocab = {k: i for i, k in enumerate(alphabet)}
            self.vocab['ï'] = self.vocab['i']
            self.vocab['¿'] = self.vocab['?']
            self.vocab['½'] = self.vocab[' ']
            self.vocab_len = len(self.vocab) - 3
            self.split = list
        self.avail_classes = []
        with open(os.path.join(dataset_dir, avail_class_fn), 'r') as avcls:
            while True:
                line = avcls.readline()
                if not line:
                    break
                self.avail_classes.append(line.strip())

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.text_cutoff = text_cutoff
        self.device = device
        if 'text_emb_dir' in kwargs:
            self.text_emb_dir = kwargs['text_emb_dir']

        if 'minibatch_size' in kwargs and kwargs['minibatch_size'] > 1:
            self.minibatch_size = min(kwargs['minibatch_size'], len(self.avail_classes))
        else:
            self.minibatch_size = len(self.avail_classes)


    def get_captions(self):

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            txt_fns = os.listdir(os.path.join(self.dataset_dir, self.text_dir, clas))

            txt_fns = list(filter(lambda s: os.path.splitext(s)[1] == '.h5' and os.path.splitext(s)[1].isdigit(), txt_fns))

            clas_txts = torch.empty(len(txt_fns), 10, self.vocab_len,
            	                    self.text_cutoff, device=self.device)

            for i, txt_fn in enumerate(txt_fns):
                txtvals = h5py.File(os.path.join(self.dataset_dir, self.text_dir, clas, txt_fn), 'r').values()

                for j, txt in enumerate(txtvals):
                    clas_txts[i, j] = self.process_text(txt)

            yield clas_txts, lbl


    def get_captions_and_image(self):
        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            txt_fns = os.listdir(os.path.join(self.dataset_dir, self.text_dir, clas))
            txt_fns = list(filter(lambda s: os.path.splitext(s)[1] == '.h5' and os.path.splitext(s)[0].isdigit(), txt_fns))

            img_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                keys = list(h5fp.keys())
                imgs = h5fp

                for i, txt_fn in enumerate(txt_fns):
                    txtvals = h5py.File(os.path.join(self.dataset_dir, self.text_emb_dir, clas, txt_fn), 'r').values()

                    clas_txts = torch.empty(len(txtvals), 1024, device=self.device)
                    for j, txt in enumerate(txtvals):
                        clas_txts[j] = torch.from_numpy(np.array(txt, dtype=np.float32)).to(self.device)
                    img = torch.from_numpy(np.array(imgs[str(i)], dtype=np.float32).transpose(1, 0)).to(self.device)

                    yield clas_txts, img, clas, i, lbl


    def get_captions_and_info(self):
        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            txt_fns = os.listdir(os.path.join(self.dataset_dir, self.text_dir, clas))
            txt_fns = list(filter(lambda s: os.path.splitext(s)[1] == '.h5' and os.path.splitext(s)[0].isdigit(), txt_fns))

            for i, txt_fn in enumerate(txt_fns):
                txtvals = h5py.File(os.path.join(self.dataset_dir, self.text_dir, clas, txt_fn), 'r').values()

                clas_txts = torch.empty(len(txtvals), self.vocab_len, self.text_cutoff, device=self.device)
                for j, txt in enumerate(txtvals):
                    clas_txts[j] = self.process_text(txt)

                yield clas_txts, clas, i, lbl


    def get_images(self):

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            img_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                imgs = torch.empty(len(h5fp.keys()), 1024, device=self.device)
                for i, key in enumerate(h5fp.keys()):
                    imgs[i] = torch.tensor(h5fp[key], device=self.device).squeeze()

            yield imgs, lbl


    def get_next_minibatch(self, n_txts=1):

        assert 1 <= n_txts <= 10

        imgs = torch.empty(self.minibatch_size, 1024, device=self.device)
        txts = torch.empty(self.minibatch_size, n_txts, self.vocab_len, self.text_cutoff, device=self.device)
        lbls = torch.empty(self.minibatch_size, dtype=torch.int, device=self.device)
        
        rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]
        for i, class_ind in enumerate(rand_class_ind):
            clas = self.avail_classes[class_ind]

            lbl = int(clas.split('.')[0])

            img_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                keys = list(h5fp.keys())
                rand_img = torch.randint(len(keys), (1,)).item()
                rand_key = keys[rand_img]
                rand_crop = torch.randint(10, (1,)).item()
                img = h5fp[rand_key][:, rand_crop]


            txt_fn = os.path.join(self.dataset_dir, self.text_dir, clas, rand_key + '.h5')
            rand_txts = torch.randperm(10)[:n_txts] + 1
            with h5py.File(txt_fn, 'r') as txtobj:
                for j, rand_txt in enumerate(rand_txts):
                    txt = txtobj['txt' + str(rand_txt.item())]
                    txt = self.process_text(txt)
                    txts[i, j] = txt

            imgs[i] = torch.tensor(img, device=self.device)
            lbls[i] = lbl

        return imgs, txts.squeeze(), lbls


    def process_text(self, text):
        text = self.split(''.join(map(chr, text[:self.text_cutoff].astype(int))))
        res = torch.zeros(self.vocab_len, self.text_cutoff, device=self.device)
        res[[[self.vocab[tok] for tok in text], range(len(text))]] = 1

        return res


    def name(self):
        return "CUBDataset"


class CUBTextEmbeddingDatasetLazy(data.Dataset):
    
    def __init__(self, dataset_dir: str, avail_class_fn: str, image_dir: str, text_dir: str, device='cuda:0', **kwargs):

        super().__init__()
        self.vocab_len = 70

        self.avail_classes = []
        with open(os.path.join(dataset_dir, avail_class_fn), 'r') as avcls:
            while True:
                lin = avcls.readline()
                if not line:
                    break
                self.avail_classes.append(line.strip())

        self.dataset_dir = dataset_dir
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.device = device

        if 'minibatch_size' in kwargs and kwargs['minibatch_size'] > 1:
            self.minibatch_size = min(kwargs['minibatch_size'], len(self.avail_classes))
        else:
            self.minibatch_size = len(self.avail_classes)

    
    def get_captions(self):

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            txt_fn = os.path.join(self.dataset_dir, self.text_dir, clas + '.t7')
            txt_np = torchfile.load(txt_fn)
            txt_t  = self.process_text(txt_np)

            yield txt_t, lbl


    def get_image(self):

        for clas in self.avail_classes:
            lbl = int(clas.split('.')[0])

            imgs_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.t7')
            imgs_np = torchfile.load(imgs_fn)

            imgs_t = torch.tensor(imgs_np[...,0], dtype=torch.float, device=self.device)
            
            yield imgs_t, lbl


    def get_next_minibatch(self, n_txts=1):

        assert 1 <= n_txts <= 10

        imgs = torch.empty(self.minibatch_size, 1024, device=self.device)
        txts = torch.empty(self.minibatch_size, ntxts, self.vocab_len, 201, device=self.device)
        lbls = torch.empty(self.minibatch_size, dtype=torch.int, device=self.device)

        rand_class_ind = torch.randperm(len(self.avail_classes))[:self.minibatch_size]

        for i, class_ind in enumerate(rand_class_ind):
            clas = self.avail_classes[class_ind]

            lbl = int(clas.split('.')[0])

            img_fn = os.path.join(self.dataset_dir, self.image_dir, clas + '.h5')
            with h5py.File(img_fn, 'r') as h5fp:
                rand_img = str(torch.randint(len(h5fp), (1,)).item())
                rand_crop = torch.randint(10, (1,)).item()
                imgs[i] = torch.tensor(h5fp[rand_img][...,rand_crop], device=self.device)

            txt_fn = os.path.join(self.dataset_dir, self.text_dir, clas + '.h5')
            with h5py.File(txt_fn, 'r') as h5fp:
                rand_txts = torch.randperm(10)[:n_txts]
                txts[i] = self.process_text(h5fp[rand_img][..., rand_txts].reshape(1, 201, len(rand_txts)))

            lbls[i] = lbl

        return imgs, txts.squeeze(), lbls


    def process_text(self, text):
        ohvec = torch.zeros(text.shape[0], text.shape[2], self.vocab_len, text.shape[1], device=self.device)
        for corr_img in range(text.shape[0]):
            for cap in range(text.shape[2]):
                for tok in range(text.shape[1]):
                    ohvec[corr_img, cap, int(text[corr_img, tok, cap]) - 1, tok] = 1

        return ohvec


if __name__ == '__main__':
    
    cub_img_text = CUBTextEmbeddingDataset(dataset_dir='../../data/cvpr2016_cub', avail_class_fn='allclasses.txt',
    	image_dir='images_h5', text_dir='text_c10', text_cutoff=80, device='cpu', text_emb_dir='text_emb1024')
    # imgs, txts, lbls = cub_img_text.get_next_minibatch(n_txts=3)
    # print(imgs.shape) # 200 x 1024
    # print(txts.shape) # 200 x 3 x 69 x 80
    # print(lbls.shape) # 200
    # for clas_txts, clas, i, lbl in cub_img_text.get_captions_and_info():
    #     print(clas_txts.shape)
    #     print(clas)
    #     print(i)
    #     print(lbl)
    for clas_txts, img, clas, i, lbl in cub_img_text.get_captions_and_image():
        print(clas_txts.shape)
        print(img.shape)