import os
import sys
sys.path.append('..')
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.text_embedding.cub_emb_dataset import CUBTextEmbeddingDataset as CUBDataset
from datasets.text_embedding.flower_emb_dataset import FlowerTextEmbeddingDataset as FlowerDataset
from datasets.text_embedding.coco_emb_dataset import COCOTextEmbeddingDataset as COCODataset
from models.encoders.text_encoder import HybridCNN
from utils.config import cfg
from utils.summary import AverageMeter, LogSummary
from utils.timer import Timer, calculate_eta


def save_model(model, trainset, step):
    save_path = os.path.join(cfg.TRAIN.SAVE_DIR, model.name() + '_LAZY', trainset.name())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = "CRNN_" + str(trainset.vocab_len)
    for channel in cfg.MODEL.CRNN.CONV_CHANNELS:
        model_name += '_' + str(channel)

    for kernel in cfg.MODEL.CRNN.CONV_KERNELS:
        model_name += '_' + str(kernel)

    for stride in cfg.MODEL.CRNN.CONV_STRIDES:
        model_name += '_' + str(stride)

    if cfg.MODEL.CRNN.RNN_BIDIR:
        model_name += '_Bidir'

    if cfg.MODEL.CRNN.LSTM:
        model_name += '_LSTM'
    else:
        model_name += '_RNN'

    model_name += '_' + str(cfg.MODEL.CRNN.RNN_HIDDEN_SIZE)
    model_name += '_' + str(cfg.MODEL.CRNN.RNN_NUM_LAYERS)
    model_name += '_' + str(step)
    print('Saving model to path {}'.format(os.path.join(save_path, model_name + '.pth')))
    torch.save(model.state_dict(), os.path.join(save_path, model_name + '.pth'))
    


def train(text_encoder, dataloader, optimizer):
    
    # if cfg.TRAIN.LR_DECAY:
    #     lr_decay = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda b: 0.98 if (b + 1) % 200 else 1)

    epoch = 0
    if cfg.TRAIN.TEXT_EMBEDDING_LAZY.RESUME_DIR:
        if not os.path.exists(cfg.TRAIN.TEXT_EMBEDDING_LAZY.RESUME_DIR):
            print("Resume file not exist.")
            exit(1)
        print("Loading pretrained model from {}".format(cfg.TRAIN.TEXT_EMBEDDING_LAZY.RESUME_DIR))
        text_encoder.load_state_dict(torch.load(cfg.TRAIN.TEXT_EMBEDDING_LAZY.RESUME_DIR))
        epoch = int(cfg.TRAIN.TEXT_EMBEDDING_LAZY.RESUME_DIR.split('_')[-1].split('.')[0]) + 1

    global_mean_loss = AverageMeter()
    log_dir = os.path.join(cfg.TRAIN.LOG_DIR, text_encoder.name() + '_LAZY', dataloader.dataset.name(), datetime.now().strftime('%b%d_%H-%M-%S_'))
    logger = LogSummary(log_dir)
    timer = Timer()
    timer.start()

    criterion = nn.SmoothL1Loss(reduction='mean')

    global_step = epoch * len(dataloader.dataset) // cfg.BATCH_SIZE

    generator_lr = cfg.TRAIN.LEARNING_RATE

    while epoch < cfg.TRAIN.TEXT_EMBEDDING_LAZY.MAX_EPOCH:

        if (epoch + 1) % 200 == 0:
            generator_lr = generator_lr * 0.7
            for param_group in optimizer.param_groups:
                param_group['lr'] = generator_lr

        for step, data in enumerate(dataloader, 0):
            global_step += 1

            captions, embeddings = data
            captions = captions.to(cfg.DEVICE)
            embeddings = embeddings.to(cfg.DEVICE)

            pred_embs = text_encoder(captions)

            loss = criterion(pred_embs, embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_mean_loss.update(loss.item())

            if global_step % cfg.TRAIN.TEXT_EMBEDDING_LAZY.LOG_INTERVAL == 0:
                speed = cfg.TRAIN.TEXT_EMBEDDING.LOG_INTERVAL / timer.elapsed_time()

                print(("epoch: {}, lr: {:.5f}, step: {}, global loss: {:.4f}, batch loss: {:.4f}, speed: {:.4f} batch /sec")
                	.format(epoch, generator_lr, global_step + 1, global_mean_loss.avg, loss.item(), speed))
                
                logger.write_scalars({'avg_loss': global_mean_loss.avg,}, 
                	tag='train', n_iter=global_step)

                sys.stdout.flush()

                timer.restart()

        if epoch % cfg.TRAIN.TEXT_EMBEDDING_LAZY.SAVE_INTERVAL == 0:
            save_model(text_encoder, dataloader.dataset, epoch)

        epoch += 1

    print("Done!")
    save_model(text_encoder, dataloader.dataset, 'final')





def main():
    
    if cfg.TRAIN.TEXT_EMBEDDING_LAZY.DATASET == 'cub':
        trainset = CUBDataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_LAZY.HDGAN.CUB.DATASET_DIR,
                              captions_dir=cfg.DATASET.TEXT_EMBEDDING_LAZY.HDGAN.CUB.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_LAZY.TEXT_CUTOFF)
    elif cfg.TRAIN.TEXT_EMBEDDING_LAZY.DATASET == 'flower':
        trainset = FlowerDataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_LAZY.HDGAN.FLOWER.DATASET_DIR,
                              captions_dir=cfg.DATASET.TEXT_EMBEDDING_LAZY.HDGAN.FLOWER.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_LAZY.TEXT_CUTOFF)
    elif cfg.TRAIN.TEXT_EMBEDDING_LAZY.DATASET == 'coco':
        trainset = COCODataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_LAZY.HDGAN.COCO.DATASET_DIR,
                              captions_dir=cfg.DATASET.TEXT_EMBEDDING_LAZY.HDGAN.COCO.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_LAZY.TEXT_CUTOFF)
    
    text_encoder = HybridCNN(vocab_dim=trainset.vocab_len, conv_channels=cfg.MODEL.CRNN.CONV_CHANNELS,
                             conv_kernels=cfg.MODEL.CRNN.CONV_KERNELS, conv_strides=cfg.MODEL.CRNN.CONV_STRIDES,
                             rnn_bidir=cfg.MODEL.CRNN.RNN_BIDIR, conv_dropout=cfg.MODEL.CRNN.CONV_DROPOUT,
                             lin_dropout=cfg.MODEL.CRNN.LIN_DROPOUT, rnn_dropout=cfg.MODEL.CRNN.RNN_DROPOUT,
                             rnn_hidden_size=cfg.MODEL.CRNN.RNN_HIDDEN_SIZE // (1 + int(cfg.MODEL.CRNN.RNN_BIDIR)),
                             rnn_num_layers=cfg.MODEL.CRNN.RNN_NUM_LAYERS, lstm=cfg.MODEL.CRNN.LSTM)\
                                .to(cfg.DEVICE).train()
    
    dataloader = DataLoader(dataset=trainset, batch_size=cfg.BATCH_SIZE,  drop_last=True, 
        shuffle=True, num_workers=int(cfg.TRAIN.WORKERS))

    optimizer = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    train(text_encoder, dataloader, optimizer)


if __name__ == '__main__':
    main()