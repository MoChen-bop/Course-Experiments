import os
import sys
sys.path.append('..')
from datetime import datetime

import torch
import torch.optim as optim

from datasets.cub_dataset import CUBTextEmbeddingDataset as CUBDataset
from datasets.flower_dataset import FlowerTextEmbeddingDataset as FlowerDataset
from models.encoders.text_encoder import HybridCNN
from models.loss.text_embedding_loss import joint_embedding_loss, Fvt
from utils.config import cfg
from utils.summary import AverageMeter, LogSummary
from utils.timer import Timer, calculate_eta


def save_model(model, trainset, step):
    save_path = os.path.join(cfg.TRAIN.SAVE_DIR, model.name(), trainset.name())
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
    


def train(text_encoder, trainset, optimizer):
    
    # if cfg.TRAIN.LR_DECAY:
    #     lr_decay = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda b: 0.98 if (b + 1) % 200 else 1)

    step = 0
    if cfg.TRAIN.TEXT_EMBEDDING.RESUME_DIR:
        if not os.path.exists(cfg.TEXT_EMBEDDING.TRAIN.RESUME_DIR):
            print("Resume file not exist.")
            exit(1)
        print("Loading pretrained model from {}".format(cfg.TRAIN.TEXT_EMBEDDING.RESUME_DIR))
        text_encoder.load_state_dict(torch.load(cfg.TRAIN.TEXT_EMBEDDING.RESUME_DIR))
        step = int(cfg.TRAIN.TEXT_EMBEDDING.RESUME_DIR.split('_')[-1].split('.')[0])

    global_mean_loss = AverageMeter()
    global_mean_accuracy = AverageMeter()
    log_dir = os.path.join(cfg.TRAIN.LOG_DIR, text_encoder.name(), trainset.name(), datetime.now().strftime('%b%d_%H-%M-%S_'))
    logger = LogSummary(log_dir)
    timer = Timer()
    timer.start()

    while step < cfg.TRAIN.TEXT_EMBEDDING.MAX_STEP:
        img_embs, txts, lbls = trainset.get_next_minibatch()
        txt_embs = text_encoder(txts)

        loss = joint_embedding_loss(img_embs, txt_embs, lbls, batched=False, device=cfg.DEVICE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_mean_loss.update(loss.item())

        # if cfg.TRAIN.LR_DECAY:
        #     lr_decay.step()

        if step % cfg.TRAIN.TEXT_EMBEDDING.LOG_INTERVAL == 0:
            speed = cfg.TRAIN.TEXT_EMBEDDING.LOG_INTERVAL / timer.elapsed_time()
            comp = Fvt(img_embs, txt_embs)
            corr = (comp.max(dim=-1)[1] == torch.arange(comp.size(0), device=cfg.DEVICE)).sum().item()
            acc = corr / comp.size(0)
            global_mean_accuracy.update(acc)

            print(("Step: {}, global loss: {:.4f}, batch loss: {:.4f}, global accuracy: {:.4f}, batch accuracy: {:.4f}, speed: {:.4f} batch /sec")
            	.format(step + 1, global_mean_loss.avg, loss.item(), global_mean_accuracy.avg, acc, speed))
            
            logger.write_scalars({'avg_loss': global_mean_loss.avg, 'avg_acc': global_mean_accuracy.avg,}, 
            	tag='train', n_iter=step)

            sys.stdout.flush()

            timer.restart()

        if step % cfg.TRAIN.TEXT_EMBEDDING.SAVE_INTERVAL == 0:
            save_model(text_encoder, trainset, step)

        step += 1

    print("Done!")
    save_model(text_encoder, trainset, 'final')





def main():
    
    if cfg.TRAIN.TEXT_EMBEDDING_DATASET == 'cub':
        trainset = CUBDataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.DATASET_DIR,
                              avail_class_fn=cfg.DATASET.TEXT_EMBEDDING_CUB.AVAIL_CLASS_FN,
                              image_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.IMAGE_DIR,
                              text_dir=cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_CUB.TEXT_CUTOFF,
                              device=cfg.DEVICE, minibatch_size=cfg.BATCH_SIZE)
    elif cfg.TRAIN.TEXT_EMBEDDING_DATASET == 'flower':
        trainset = FlowerDataset(dataset_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.DATASET_DIR,
                              avail_class_fn=cfg.DATASET.TEXT_EMBEDDING_FLOWER.AVAIL_CLASS_FN,
                              image_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.IMAGE_DIR,
                              text_dir=cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_DIR,
                              text_cutoff=cfg.DATASET.TEXT_EMBEDDING_FLOWER.TEXT_CUTOFF,
                              device=cfg.DEVICE, minibatch_size=cfg.BATCH_SIZE)
    
    text_encoder = HybridCNN(vocab_dim=trainset.vocab_len, conv_channels=cfg.MODEL.CRNN.CONV_CHANNELS,
                             conv_kernels=cfg.MODEL.CRNN.CONV_KERNELS, conv_strides=cfg.MODEL.CRNN.CONV_STRIDES,
                             rnn_bidir=cfg.MODEL.CRNN.RNN_BIDIR, conv_dropout=cfg.MODEL.CRNN.CONV_DROPOUT,
                             lin_dropout=cfg.MODEL.CRNN.LIN_DROPOUT, rnn_dropout=cfg.MODEL.CRNN.RNN_DROPOUT,
                             rnn_hidden_size=cfg.MODEL.CRNN.RNN_HIDDEN_SIZE // (1 + int(cfg.MODEL.CRNN.RNN_BIDIR)),
                             rnn_num_layers=cfg.MODEL.CRNN.RNN_NUM_LAYERS, lstm=cfg.MODEL.CRNN.LSTM)\
                                .to(cfg.DEVICE).train()

    optimizer = optim.Adam(text_encoder.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    train(text_encoder, trainset, optimizer)


if __name__ == '__main__':
    main()