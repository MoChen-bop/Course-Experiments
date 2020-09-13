import os
import sys
sys.path.append('..')
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.encoders.xception import Xception41 as Xception
from datasets.augmentation import Augmentation, BaseTransform
from datasets.coco_dataset import COCOClsDataset
from utils.config import cfg
from utils.summary import AverageMeter, LogSummary
from utils.timer import Timer


def save_model(model, trainset, epoch):
    
    save_path = os.path.join(cfg.TRAIN.SAVE_DIR, model.name(), trainset.name())
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_name = 'Xception_' + str(epoch) + '.pth'
    print("Saving model to path {}".format(os.path.join(save_path, model_name)))
    torch.save(model.state_dict(), os.path.join(save_path, model_name))



def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / cfg.BATCH_SIZE))

    return res 


def train(model, dataloader, optimizer):
    
    epoch = 0
    if cfg.TRAIN.XCEPTION.RESUME_DIR:
        if not os.path.exists(cfg.TRAIN.XCEPTION.RESUME_DIR):
            print("Resume file not exist.")
            exit(1)
        print("Loading pretrained model from {}".format(cfg.TRAIN.XCEPTION.RESUME_DIR))
        model.load_state_dict(torch.load(cfg.TRAIN.XCEPTION.RESUME_DIR))
        epoch = int(cfg.TRAIN.XCEPTION.RESUME_DIR.split('_')[-1].split('.')[0]) + 1

    global_mean_loss = AverageMeter()
    global_mean_accuracy_top1 = AverageMeter()
    global_mean_accuracy_top5 = AverageMeter()

    log_dir = os.path.join(cfg.TRAIN.LOG_DIR, model.name(), dataloader.dataset.name(),
        datetime.now().strftime('%b%d_%H-%M-%S_'))
    logger = LogSummary(log_dir)

    timer = Timer()
    timer.start()
    
    model = model.to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss().to(cfg.DEVICE)
    model.train()
    global_step = int(epoch * cfg.DATASET.COCO.SIZE / cfg.BATCH_SIZE)

    while epoch < cfg.TRAIN.XCEPTION.MAX_STEP:

        for step, (image, label) in enumerate(dataloader):

            optimizer.zero_grad()
            image = image.to(cfg.DEVICE)
            label = label.to(cfg.DEVICE)

            logit = model(image)
            loss = criterion(logit, label)

            loss.backward()
            optimizer.step()

            global_mean_loss.update(loss.item())

            precise_top1, precise_top5 = accuracy(logit, label, topk=(1, 5))
            global_mean_accuracy_top1.update(precise_top1.item())
            global_mean_accuracy_top5.update(precise_top5.item())

            if global_step % cfg.TRAIN.XCEPTION.LOG_INTERVAL == 0:
                speed = cfg.TRAIN.XCEPTION.LOG_INTERVAL / timer.elapsed_time()
                print(("Epoch: {}, Step: {}/{}, global loss: {:.4f}, batch_loss: {:.4f}, global precise@1: {:.2f},"
                	" batch_precise@1: {:.2f}, global precise@5: {:.2f}, batch_precise@5: {:.2f}, speed: {:.2f} step /sec"
                	.format(epoch, step, (global_step), global_mean_loss.avg, loss.item(), global_mean_accuracy_top1.avg, precise_top1.item(),
                	global_mean_accuracy_top5.avg, precise_top5.item(), speed)))

                logger.write_scalars({'avg_loss': global_mean_loss.avg, 'avg_p@1': global_mean_accuracy_top1.avg,
                	'avg_p@5': global_mean_accuracy_top5.avg, }, tag='train', n_iter=global_step)

                sys.stdout.flush()

                timer.restart()

            global_step += 1

        if epoch % cfg.TRAIN.XCEPTION.SAVE_INTERVAL == 0:
        	save_model(model, dataloader.dataset, epoch)

        epoch += 1

    save_model(model, dataloader.dataset, 'final')




def main():
    transform = Augmentation(cfg.DATASET.COCO.IMAGE_SIZE, cfg.AUG.IMAGE_MEAN, cfg.AUG.IMAGE_STD)
    dataset = COCOClsDataset(cfg.DATASET.COCO.TRAIN_IMAGE_DIR, cfg.DATASET.COCO.TRAIN_IMAGE_LABEL_DIR,
    	cfg.DATASET.COCO.TRAIN_INSTANCES_DIR, transform)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, 
    	num_workers=cfg.DATASET.DATA_FEEDER_NUM)

    model = Xception(cfg.DATASET.COCO.CATEGORIES)

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)

    train(model, dataloader, optimizer)


if __name__ == '__main__':
    main()