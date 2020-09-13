import os
import sys
sys.path.append('..')
import numpy as np 
from PIL import Image 
from utils.config import cfg
import cv2

def vis_fake_real_image(real_imgs, fake_imgs, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    batch_size = real_imgs.shape[0]

    for i in range(batch_size):
        real_img = real_imgs[i].transpose((1, 2, 0))
        fake_img = fake_imgs[i].transpose((1, 2, 0))
        real_img = real_img * cfg.AUG.IMAGE_STD + cfg.AUG.IMAGE_MEAN
        fake_img = fake_img * cfg.AUG.IMAGE_STD + cfg.AUG.IMAGE_MEAN

        real_img = (real_img * 255).astype(np.uint8)
        fake_img = (fake_img * 255).astype(np.uint8)

        real_img_cv2 = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
        fake_img_cv2 = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)

        show = np.concatenate([real_img_cv2, fake_img_cv2], axis=1)
        path = os.path.join(dir, '{}.png'.format(i))
        cv2.imwrite(path, show)


def vis_fake_real_image_2(real_imgs, stage1_imgs, fake_imgs, dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    batch_size = real_imgs.shape[0]

    for i in range(batch_size):
        real_img = real_imgs[i].transpose((1, 2, 0))
        stage1_img = stage1_imgs[i].transpose((1, 2, 0))
        fake_img = fake_imgs[i].transpose((1, 2, 0))

        real_img = real_img * cfg.AUG.IMAGE_STD + cfg.AUG.IMAGE_MEAN
        stage1_img = stage1_img * cfg.AUG.IMAGE_STD + cfg.AUG.IMAGE_MEAN
        fake_img = fake_img * cfg.AUG.IMAGE_STD + cfg.AUG.IMAGE_MEAN

        real_img = (real_img * 255).astype(np.uint8)
        stage1_img = (stage1_img * 255).astype(np.uint8)
        fake_img = (fake_img * 255).astype(np.uint8)

        stage1_img = cv2.resize(stage1_img, (fake_img.shape[1], fake_img.shape[0]))

        real_img_cv2 = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
        stage1_img_cv2 = cv2.cvtColor(stage1_img, cv2.COLOR_RGB2BGR)
        fake_img_cv2 = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)

        show = np.concatenate([real_img_cv2, stage1_img_cv2, fake_img_cv2], axis=1)
        path = os.path.join(dir, '{}.png'.format(i))
        cv2.imwrite(path, show)