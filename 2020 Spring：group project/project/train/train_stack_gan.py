from __future__ import print_function
import os
import sys
sys.path.append('..')
from six.moves import range
import numpy as np 
from datetime import datetime


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from models.encoders.text_encoder import HybridCNN
from models.generators.stackgan import STAGE1_G, STAGE2_G
from models.discriminators.stackgan import STAGE1_D, STAGE2_D
from models._utils import weights_init
from models.loss.stack_gan_loss import KL_loss
from models.loss.stack_gan_loss import compute_discriminator_loss, compute_generator_loss

from datasets.image_caption.flower_lazy_dataset import FlowerTextDataset as FlowerDataset
from datasets.image_caption.cub_lazy_dataset import CUBTextDataset as CUBDataset
from datasets.augmentation import BaseTransform, Augmentation

from utils.config import cfg 
from utils.summary import AverageMeter, LogSummary
from utils.timer import Timer

from utils.visualize import vis_fake_real_image, vis_fake_real_image_2


def save_model(netG, netD, trainset, epoch):
    save_dir = os.path.join(cfg.TRAIN.SAVE_DIR, 'StackGAN', trainset.name(), netG.name())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saved_model_name = netG.name() + '_{}_{}_{}_{}.pth'.format(
    	cfg.MODEL.STACKGAN.GF_DIM, cfg.MODEL.STACKGAN.CONDITION_DIM, cfg.MODEL.STACKGAN.Z_DIM, epoch)
    print("Saving netG to path: {}.".format(os.path.join(save_dir, saved_model_name)))
    torch.save(netG.state_dict(), os.path.join(save_dir, saved_model_name))

    save_dir = os.path.join(cfg.TRAIN.SAVE_DIR, 'StackGAN', trainset.name(), netD.name())
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saved_model_name = netD.name() + '_{}_{}_{}_{}.pth'.format(
    	cfg.MODEL.STACKGAN.DF_DIM, cfg.MODEL.STACKGAN.CONDITION_DIM, cfg.MODEL.STACKGAN.Z_DIM, epoch)
    print("Saving netD to path: {}.".format(os.path.join(save_dir, saved_model_name)))
    torch.save(netD.state_dict(), os.path.join(save_dir, saved_model_name))


def load_model(model, model_path):
    if not os.path.exists(model_path):
        print("Counldn't find path {}.".format(model_path))
        exit(1)
    print("Loading model from path: {}".format(model_path))
    model.load_state_dict(torch.load(model_path))



def build_network():
    if cfg.TRAIN.STACKGAN.STAGE == 1:
        netG = STAGE1_G(t_dim=cfg.MODEL.STACKGAN.TEXT_DIM, gf_dim=cfg.MODEL.STACKGAN.GF_DIM,
        	condition_dim=cfg.MODEL.STACKGAN.CONDITION_DIM, z_dim=cfg.MODEL.STACKGAN.Z_DIM)
        netD = STAGE1_D(df_dim=cfg.MODEL.STACKGAN.DF_DIM, condition_dim=cfg.MODEL.STACKGAN.CONDITION_DIM)

        netG.apply(weights_init)
        netD.apply(weights_init)

        if cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE1_G:
            load_model(netG, cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE1_G)

        if cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE1_D:
            load_model(netD, cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE1_D)
    elif cfg.TRAIN.STACKGAN.STAGE == 2:
        stage1_G = STAGE1_G(t_dim=cfg.MODEL.STACKGAN.TEXT_DIM, gf_dim=cfg.MODEL.STACKGAN.GF_DIM,
        	condition_dim=cfg.MODEL.STACKGAN.CONDITION_DIM, z_dim=cfg.MODEL.STACKGAN.Z_DIM)
        netG = STAGE2_G(stage1_G, gf_dim=cfg.MODEL.STACKGAN.GF_DIM,
        	condition_dim=cfg.MODEL.STACKGAN.CONDITION_DIM, z_dim=cfg.MODEL.STACKGAN.Z_DIM)
        netD = STAGE2_D(df_dim=cfg.MODEL.STACKGAN.DF_DIM, condition_dim=cfg.MODEL.STACKGAN.CONDITION_DIM)

        netG.apply(weights_init)
        netD.apply(weights_init)

        if cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE1_G:
            load_model(netG.STAGE1_G, cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE1_G)

        if cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE2_G:
            load_model(netG, cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE2_G)

        if cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE2_D:
            load_model(netD, cfg.TRAIN.STACKGAN.PRETRAINED_MODEL.STAGE2_D)

    netG.to(cfg.DEVICE)
    netD.to(cfg.DEVICE)

    return netG, netD


def train(netG, netD, optimizerG, optimizerD, train_loader):
    noise = Variable(torch.FloatTensor(cfg.BATCH_SIZE, cfg.MODEL.STACKGAN.Z_DIM)).to(cfg.DEVICE)
    fixed_noise = Variable(torch.FloatTensor(cfg.BATCH_SIZE, cfg.MODEL.STACKGAN.Z_DIM).normal_(0, 1),
        volatile=True).to(cfg.DEVICE)
    real_labels = Variable(torch.FloatTensor(cfg.BATCH_SIZE).fill_(1)).to(cfg.DEVICE)
    fake_labels = Variable(torch.FloatTensor(cfg.BATCH_SIZE).fill_(0)).to(cfg.DEVICE)

    global_mean_D_loss = AverageMeter()
    global_mean_D_loss_real = AverageMeter()
    global_mean_D_loss_wrong = AverageMeter()
    global_mean_D_loss_fake = AverageMeter()
    global_mean_G_loss = AverageMeter()
    global_mean_KL_loss = AverageMeter()

    log_dir = os.path.join(cfg.TRAIN.LOG_DIR, "StackGAN", 'Stage' + str(cfg.TRAIN.STACKGAN.STAGE), train_loader.dataset.name(), 
        datetime.now().strftime('%b%d_%H-%M-%S_'))
    logger = LogSummary(log_dir)
    timer = Timer()
    timer.start()

    epoch = cfg.TRAIN.STACKGAN.BEGIN_EPOCH
    global_step = epoch * len(train_loader.dataset) // cfg.BATCH_SIZE
    for epoch in range(epoch, cfg.TRAIN.STACKGAN.MAX_EPOCH):

        #if epoch % cfg.TRAIN.STACKGAN.DECAY_EPOCH == 0:
        generator_lr = cfg.TRAIN.STACKGAN.GENERATOR_LR * np.power(0.5, epoch // cfg.TRAIN.STACKGAN.DECAY_EPOCH)
        for param_group in optimizerG.param_groups:
            param_group['lr'] = generator_lr
        discriminator_lr = cfg.TRAIN.STACKGAN.DISCRIMINATOR_LR * np.power(0.5, epoch // cfg.TRAIN.STACKGAN.DECAY_EPOCH)
        for param_group in optimizerD.param_groups:
            param_group['lr'] = discriminator_lr
        print("Epoch {}, generator_lr: {:.6f}, discriminator_lr: {:.6f}".format(epoch, generator_lr, discriminator_lr))

        for step, data in enumerate(train_loader, 0):
            global_step += 1
            real_img_cpu, txt_embedding = data
            real_imgs = real_img_cpu.to(cfg.DEVICE)
            txt_embedding = txt_embedding.to(cfg.DEVICE)

            noise.data.normal_(0, 1)
            if cfg.TRAIN.STACKGAN.STAGE == 1:
                _, fake_imgs, mu, logvar = netG(txt_embedding, noise)
            else:
                stage1_imgs, fake_imgs, mu, logvar = netG(txt_embedding, noise) 

            netD.zero_grad()
            errD, errD_real, errD_wrong, errD_fake = \
                compute_discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, mu)

            errD.backward()
            optimizerD.step()

            netG.zero_grad()
            errG = compute_generator_loss(netD, fake_imgs, real_labels, mu)

            kl_loss = KL_loss(mu, logvar)
            errG_total = errG + kl_loss * cfg.TRAIN.STACKGAN.COEFF_KL
            errG_total.backward()
            optimizerG.step()

            global_mean_D_loss.update(errD.item())
            global_mean_D_loss_real.update(errD_real)
            global_mean_D_loss_wrong.update(errD_wrong)
            global_mean_D_loss_fake.update(errD_fake)
            global_mean_G_loss.update(errG.item())
            global_mean_KL_loss.update(kl_loss.item())

            if global_step % cfg.TRAIN.STACKGAN.LOG_INTERVAL == 0:
                speed = cfg.TRAIN.STACKGAN.LOG_INTERVAL / timer.elapsed_time()

                print(("Epoch: {}, Step:{}[{}], Loss_D: {:.4f}[{:.4f}], Loss_G: {:.4f}[{:.4f}],"
                	" Loss_real: {:.4f}[{:.4f}],  Loss_wrong: {:.4f}[{:.4f}], Loss_fake: {:.4f}[{:.4f}],"
                	" Speed: {:.4f} step / second".format(epoch, global_step, step, global_mean_D_loss.avg, errD.item(),
                	   global_mean_G_loss.avg, errG.item(), global_mean_D_loss_real.avg, errD_real,
                	   global_mean_D_loss_wrong.avg, errD_wrong, global_mean_D_loss_fake.avg, errD_fake,
                	   speed)))

                logger.write_scalars({'Loss_D': global_mean_D_loss.avg, 'Loss_G': global_mean_G_loss.avg, 'Loss_wrong': global_mean_D_loss_wrong.avg,
                    'Loss_fake': global_mean_D_loss_fake.avg, 'Loss_real': global_mean_D_loss_real.avg}, tag='train', n_iter=global_step)

                sys.stdout.flush() 
                timer.restart()

            if global_step % cfg.TRAIN.STACKGAN.VIS_INTERVAL == 0:
                if cfg.TRAIN.STACKGAN.STAGE == 1:
                    vis_fake_real_image(real_imgs.detach().cpu().numpy(), fake_imgs.detach().cpu().numpy(), os.path.join(cfg.TRAIN.VIS_DIR, 
                        'StackGAN', 'Stage' + str(cfg.TRAIN.STACKGAN.STAGE), cfg.TRAIN.STACKGAN.DATASET))
                else:
                    vis_fake_real_image_2(real_imgs.detach().cpu().numpy(), stage1_imgs.detach().cpu().numpy(), 
                        fake_imgs.detach().cpu().numpy(), os.path.join(cfg.TRAIN.VIS_DIR, 
                        'StackGAN', 'Stage' + str(cfg.TRAIN.STACKGAN.STAGE), cfg.TRAIN.STACKGAN.DATASET))


        if epoch % cfg.TRAIN.STACKGAN.SAVE_INTERVAL == 0:
            save_model(netG, netD, train_loader.dataset, epoch)

    print("Done!")
    save_model(netG, netD, train_loader.dataset, "final")


def main():

    if cfg.TRAIN.STACKGAN.DATASET == 'cub':
        transform = Augmentation(cfg.DATASET.STACKGAN.IMAGE_SIZE, cfg.AUG.IMAGE_MEAN, cfg.AUG.IMAGE_STD)
        trainset = CUBDataset(image_dir=cfg.DATASET.CUB.IMAGE_DIR, embedding_dir=cfg.DATASET.CUB.TEXT_EMBEDDING_DIR,
            transform=transform, device=cfg.DEVICE,)
    elif cfg.TRAIN.STACKGAN.DATASET == 'flower':
        transform = BaseTransform(cfg.DATASET.STACKGAN.IMAGE_SIZE, cfg.AUG.IMAGE_MEAN, cfg.AUG.IMAGE_STD)
        trainset = FlowerDataset(image_dir=cfg.DATASET.FLOWER.IMAGE_DIR, embedding_dir=cfg.DATASET.FLOWER.TEXT_EMBEDDING_DIR,
            transform=transform, device=cfg.DEVICE,)

    dataloader = DataLoader(dataset=trainset, batch_size=cfg.BATCH_SIZE, drop_last=True, 
    	shuffle=True, num_workers=int(cfg.TRAIN.WORKERS))
    netG, netD = build_network()

    optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN.STACKGAN.DISCRIMINATOR_LR, betas=(0.5, 0.999))

    netG_para = []
    for p in netG.parameters():
        if p.requires_grad:
            netG_para.append(p)
    optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN.STACKGAN.GENERATOR_LR, betas=(0.5, 0.999))

    train(netG, netD, optimizerG, optimizerD, dataloader)


if __name__ == '__main__':
    main()