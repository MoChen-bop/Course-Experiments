from __future__ import print_function

import argparse
import logging
import os
import sys
sys.path.append('..')
import six
import time
import random
import numpy as np

import paddle
import paddle.fluid as fluid

from models.dnn import DNN
from utils.config import config as cfg
from utils.config import print_config, update_config
from utils.option import BaseOptions


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def get_dataset(inputs):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python ../datasets/criteo_dataset_generator.py")
    dataset.set_batch_size(cfg.batch_size)
    thread_num = int(cfg.cpu_num)
    file_list = [
        os.path.join(cfg.train_files_path, x) for x in os.listdir(cfg.train_files_path)
    ]
    #logger.info("file list: {}".format(file_list))
    return dataset, file_list


def train():
    if cfg.train_model == 'dnn':
        model = DNN()
    
    inputs = model.input_data()
    avg_cost, auc_var = model.net(inputs)
    
    optimizer = fluid.optimizer.Adam(cfg.learning_rate)
    optimizer.minimize(avg_cost)
    
    place = fluid.CUDAPlace(0) if cfg.use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    dataset, file_list = get_dataset(inputs)

    logger.info("Training Begin")
    for epoch in range(cfg.epoches):
        random.shuffle(file_list)
        dataset.set_filelist(file_list)

        start_time = time.time()
        exe.train_from_dataset(program=fluid.default_main_program(),
                               dataset=dataset,
                               fetch_list=[avg_cost, auc_var],
                               fetch_info=['Epoch {} cost: '.format(epoch + 1), ' - auc: '],
                               print_period=cfg.log_interval,
                               debug=False)
        end_time = time.time()
        logger.info("epoch %d finished, use time = %ds \n" % ((epoch + 1), end_time - start_time))

        if (epoch + 1) % cfg.save_interval == 0:
            model_path = os.path.join(str(cfg.save_path), model.name, model.name + "_epoch_" + str(epoch + 1))
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            logger.info("saving model to %s \n" % (model_path))
            fluid.save(fluid.default_main_program(), os.path.join(model_path, "checkpoint"))
    logger.info("Done.")


def main():
    train()


if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()