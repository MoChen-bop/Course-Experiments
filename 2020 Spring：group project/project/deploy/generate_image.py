import os
import sys
sys.path.append('..')
import numpy as np 
import argparse
import cv2
import pickle
from PIL import Image

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.encoders.text_encoder import HybridCNN as CRNN
from models.generators.icml import icml_G as SingleGAN
from models.generators.stackgan import STAGE1_G as StackGAN1
from models.generators.stackgan import STAGE2_G as StackGAN2
from models.generators.hdgan import Generator as HDGAN 

from utils.config import cfg 


def load_model(model, path):
    if not os.path.exists(path):
        print("Counldn't find path {}".format(path))
        exit(1)
    print("Loading model from path: {}".format(path))
    weights_dict = torch.load(path, map_location=lambda storage, loc: storage)
    sample_weight_name = [a for a in weights_dict.keys()][0]
    if 'module' in sample_weight_name: # if the saved model is wrapped by DataParallel. 
        model = nn.parallel.DataParallel(model, device_ids=[0])
    model.load_state_dict(weights_dict, strict=False)


def process_caption(caption):
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:\'"/\\|_@#$%^&*~`+-=<>()[]{}'
    vocab = {k: i for i, k in enumerate(alphabet)}
    vocab['ï'] = vocab['i']
    vocab['¿'] = vocab['?']
    vocab['½'] = vocab[' ']
    vocab_len = len(vocab) - 3
    onehot = np.zeros((vocab_len, cfg.DEPLOY.TEXT_CUTOFF))
    onehot[tuple([[vocab[tok] for tok in caption], range(len(caption))])] = 1
    onehot = np.array([onehot])
    onehot = torch.from_numpy(onehot.astype(np.float32))
    return onehot


def visualize(generated_images, size):
    if not os.path.exists(cfg.DEPLOY.VIS_DIR):
        os.makedirs(cfg.DEPLOY.VIS_DIR)
    for i, image in enumerate(generated_images):
        image = image.detach().cpu().numpy()[0]

        image = image.transpose((1, 2, 0))
        # image = image * cfg.AUG.IMAGE_STD + cfg.AUG.IMAGE_MEAN
        # image = image * 255
        image = (image + 1.0) * 127.5
        image = image.astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        path = os.path.join(cfg.DEPLOY.VIS_DIR, '{}.jpg'.format(i))
        cv2.imwrite(path, image)


def generate_image(text_encoder, model, caption_tensor, size, n):
    with torch.no_grad():
        noise = Variable(torch.FloatTensor(1, cfg.DEPLOY.NOISE_DIM)).to(cfg.DEPLOY.DEVICE)
        generated_images = []
        for _ in range(n):
            caption_emb = text_encoder(caption_tensor)
            # caption_emb = caption_tensor
            noise.data.normal_(0, 1)
            fake_image = model(caption_emb, noise)
            generated_images.append(fake_image)

    visualize(generated_images, size)



def main(args):
    
    text_encoder = CRNN(vocab_dim=cfg.DEPLOY.VOCAB_DIM, conv_channels=cfg.DEPLOY.CRNN.CONV_CHANNELS,
                             conv_kernels=cfg.DEPLOY.CRNN.CONV_KERNELS, conv_strides=cfg.DEPLOY.CRNN.CONV_STRIDES,
                             rnn_bidir=cfg.DEPLOY.CRNN.RNN_BIDIR, conv_dropout=cfg.DEPLOY.CRNN.CONV_DROPOUT,
                             lin_dropout=cfg.DEPLOY.CRNN.LIN_DROPOUT, rnn_dropout=cfg.DEPLOY.CRNN.RNN_DROPOUT,
                             rnn_hidden_size=cfg.DEPLOY.CRNN.RNN_HIDDEN_SIZE // (1 + int(cfg.DEPLOY.CRNN.RNN_BIDIR)),
                             rnn_num_layers=cfg.DEPLOY.CRNN.RNN_NUM_LAYERS, lstm=cfg.DEPLOY.CRNN.LSTM)\
                                .to(cfg.DEPLOY.DEVICE).eval()
    if args.model == 'SingleGAN':
        model = SingleGAN(cfg.DEPLOY.SINGLEGAN.NC, cfg.DEPLOY.SINGLEGAN.NGF, deploy=True)
        load_model(model, cfg.DEPLOY.SINGLE_GAN.PRETRAINED_MODEL)
    elif args.model == 'StackGAN':
        stage1 = StackGAN1(t_dim=cfg.DEPLOY.TEXT_EMBEDDING_DIM, gf_dim=cfg.DEPLOY.STACKGAN.GF_DIM,
        	condition_dim=cfg.DEPLOY.STACKGAN.CONDITION_DIM, z_dim=cfg.DEPLOY.STACKGAN.Z_DIM)
        model = StackGAN2(stage1, gf_dim=cfg.DEPLOY.STACKGAN.GF_DIM,
        	condition_dim=cfg.DEPLOY.STACKGAN.CONDITION_DIM, z_dim=cfg.DEPLOY.STACKGAN.Z_DIM, deploy=True,
            stage2=(args.domain != 'COCO'))
        if args.domain == 'COCO':
            load_model(text_encoder, cfg.DEPLOY.CRNN.STACK_GAN.COCO.PRETRAINED_MODEL)
            load_model(model, cfg.DEPLOY.STACK_GAN.COCO.PRETRAINED_MODEL)
        elif args.domain == 'Flower':
            load_model(text_encoder, cfg.DEPLOY.CRNN.HDGAN.FLOWER.PRETRAINED_MODEL)
            load_model(model, cfg.DEPLOY.STACK_GAN.FLOWER.PRETRAINED_MODEL)            
    elif args.model == 'HDGAN':
        model = HDGAN(sent_dim=cfg.DEPLOY.TEXT_EMBEDDING_DIM, noise_dim=cfg.DEPLOY.NOISE_DIM, 
        	emb_dim=cfg.DEPLOY.HDGAN.EMB_DIM, hid_dim=cfg.DEPLOY.HDGAN.HID_DIM, 
        	num_resblock=cfg.DEPLOY.HDGAN.NUM_RESBLOCK, deploy=True)
        if args.domain == 'CUB':
            load_model(text_encoder, cfg.DEPLOY.CRNN.HDGAN.CUB.PRETRAINED_MODEL)
            load_model(model, cfg.DEPLOY.HDGAN.CUB.PRETRAINED_MODEL)
        elif args.domain == 'Flower':
            load_model(text_encoder, cfg.DEPLOY.CRNN.HDGAN.FLOWER.PRETRAINED_MODEL)
            load_model(model, cfg.DEPLOY.HDGAN.FLOWER.PRETRAINED_MODEL)
        elif args.domain == 'COCO':
            load_model(text_encoder, cfg.DEPLOY.CRNN.HDGAN.COCO.PRETRAINED_MODEL)
            load_model(model, cfg.DEPLOY.HDGAN.COCO.PRETRAINED_MODEL)
    
    model = model.to(cfg.DEPLOY.DEVICE)
    model.eval()

    caption_tensor = process_caption(args.caption.lower().strip())
    caption_tensor = caption_tensor.to(cfg.DEPLOY.DEVICE)


    # with open(os.path.join(cfg.DATASET.TEXT_EMBEDDING_LAZY.STACKGAN.COCO.DATASET_DIR, 
    #     'char-CNN-RNN-embeddings.pickle'), 'rb') as f:
    #     caption_embedding = pickle.load(f, encoding='latin')
    # caption_tensor = torch.from_numpy(caption_embedding[0][:1]).to(cfg.DEPLOY.DEVICE)

    generate_image(text_encoder, model, caption_tensor, args.size, args.n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')
    parser.add_argument('--model', type=str, choices=['SingleGAN', 'StackGAN', 'HDGAN'], default='HDGAN', help='which model you want to use to generate your image.')
    parser.add_argument('--domain', type=str, choices=['CUB', 'Flower', 'COCO'], default='CUB', help='the category of your image you want to generate.')
    parser.add_argument('--caption', type=str, default='a white bird with black tipped wings and a long grey beak.',
    	help='a short line text description of your image.')
    parser.add_argument('--size', type=int, default=256, help='the size of your image.')
    parser.add_argument('--n', type=int, default=8, help='how many images do you want to generate.')

    args = parser.parse_args()
    main(args)