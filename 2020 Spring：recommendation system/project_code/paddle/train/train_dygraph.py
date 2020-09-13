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
from paddle.fluid.dygraph.base import to_variable

from models.drnn import DRNN
from models.fcdnn import FCDNN
from models.dnn_dygraph import DNN
from utils.config import config as cfg
from utils.config import print_config, update_config
from utils.option import BaseOptions
from datasets.criteo_dataset_feeder_dygraph import CriteoDataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def evaluate(model, epoch):
    place = fluid.CUDAPlace(0) if cfg.use_cuda else fluid.CPUPlace()
    inference_scope = fluid.Scope()
    test_files = [
        os.path.join(cfg.evaluate_file_path, x) for x in os.listdir(cfg.evaluate_file_path)
    ]
    dataset = CriteoDataset()
    test_reader = paddle.batch(dataset.test(test_files), batch_size=cfg.batch_size)

    with fluid.dygraph.guard(place):
        model.eval()
        logger.info('Begin evaluate model.')

        run_index = 0
        infer_auc = 0.0
        L = []
        for batch_id, data in enumerate(test_reader()):
            dense_feature, sparse_feature, label = zip(*data)
                
            sparse_feature = np.array(sparse_feature, dtype=np.int64)
            dense_feature = np.array(dense_feature, dtype=np.float32)
            label = np.array(label, dtype=np.int64)
            sparse_feature, dense_feature, label = [
                to_variable(i)
                for i in [sparse_feature, dense_feature, label]
            ]

            avg_cost, auc_var = model(dense_feature, sparse_feature, label)

            run_index += 1
            infer_auc += auc_var.numpy().item()
            L.append(avg_cost.numpy() / cfg.batch_size)

            if batch_id % cfg.log_interval == 0:
                logger.info("TEST --> batch: {} loss: {} auc: {}".format(
                    batch_id, avg_cost.numpy() / cfg.batch_size, infer_auc / run_index))

        infer_loss = np.mean(L)
        infer_auc = infer_auc / run_index
        infer_result = {}
        infer_result['loss'] = infer_loss
        infer_result['auc'] = infer_auc
        if not os.path.isdir(os.path.join(cfg.log_dir, model.name)):
            os.makedirs(os.path.join(cfg.log_dir, model.name))
        log_path = os.path.join(cfg.log_dir, model.name, str(epoch) + '_infer_result.log')
        
        logger.info(str(infer_result))
        with open(log_path, 'w+') as f:
            f.write(str(infer_result))
        logger.info("Done.")
    return infer_result

def set_zero(var_name,
             scope=fluid.global_scope(),
             place=fluid.CPUPlace(),
             param_type="int64"):
    param = scope.var(var_name).get_tensor()
    param_array = np.zeros(param._get_dims()).astype(param_type)
    param.set(param_array, place)

def train():
    place = fluid.CUDAPlace(0) if cfg.use_cuda else fluid.CPUPlace()
    if cfg.train_model == 'drnn':
        with fluid.dygraph.guard(place):
            model = DRNN()
    elif cfg.train_model == 'dnn':
        with fluid.dygraph.guard(place):
            model = DNN()
    elif cfg.train_model == 'fcdnn':
        with fluid.dygraph.guard(place):
            model = FCDNN()
    
    with fluid.dygraph.guard(place):
        optimizer = fluid.optimizer.Adam(
                learning_rate=cfg.learning_rate,
                parameter_list=model.parameters(),)
                #regularization=fluid.regularizer.L2DecayRegularizer(cfg.reg))
        # optimizer = fluid.optimizer.SGD(learning_rate=cfg.learning_rate,
        #                                 parameter_list=model.parameters())
        dataset = CriteoDataset()
        file_list = [
            os.path.join(cfg.train_files_path, x) for x in os.listdir(cfg.train_files_path)
        ]
        train_reader = paddle.batch(dataset.test(file_list), batch_size=cfg.batch_size)
        
        start_epoch = 0
        if cfg.checkpoint:
            model_dict, optimizer_dict = fluid.dygraph.load_dygraph(
                cfg.checkpoint)
            model.set_dict(model_dict)
            optimizer.set_dict(optimizer_dict)
            start_epoch = int(
                os.path.basename(cfg.checkpoint).split("_")[
                    -1])  # get next train epoch
            logger.info("load model {} finished.".format(cfg.checkpoint))

        logger.info("Training Begin")

        for epoch in range(start_epoch, cfg.epoches):
            start_time = time.time()
            batch_begin = time.time()
            total_loss = 0.0
            total_auc = 0.0
            count = 0
            auc_metric = fluid.metrics.Auc('ROC')
            
            if not os.path.isdir(os.path.join(cfg.log_dir, model.name)):
                os.makedirs( os.path.join(cfg.log_dir, model.name))
            log_path = os.path.join(cfg.log_dir, model.name, str(epoch + 1) + '_train_result.log')
            f = open(log_path, 'w+')

            for batch_id, data in enumerate(train_reader()):
                dense_feature, sparse_feature, label = zip(*data)
                
                sparse_feature = np.array(sparse_feature, dtype=np.int64)
                dense_feature = np.array(dense_feature, dtype=np.float32)
                label = np.array(label, dtype=np.int64)
                sparse_feature, dense_feature, label = [
                    to_variable(i)
                    for i in [sparse_feature, dense_feature, label]
                ]

                avg_cost, auc_var = model(dense_feature, sparse_feature, label)

                avg_cost.backward()
                optimizer.minimize(avg_cost)
                model.clear_gradients()
                total_loss += avg_cost.numpy().item() / cfg.batch_size
                total_auc += auc_var.numpy().item()
                count += 1

                if (batch_id + 1) % cfg.log_interval == 0:
                    logger.info(
                        "epoch: %d, batch_id: %d, loss: %.6f, auc: %.6f" % (
                            epoch + 1, batch_id + 1, total_loss / count, total_auc / count))
                    batch_begin = time.time()
                
                if (batch_id + 1) % cfg.log_interval_2 == 0:
                    f.write('%d,%d,%.4f,%.4f\n' % (epoch + 1, batch_id + 1, total_loss / count, total_auc / count))

            end_time = time.time()
            logger.info("epoch %d finished, use time = %ds \n" % ((epoch + 1), end_time - start_time))

            if (epoch + 1) % cfg.save_interval == 0:
                model_path = os.path.join(str(cfg.save_path), model.name, model.name + "_epoch_" + str(epoch + 1))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                logger.info("saving model to %s \n" % (model_path))
                fluid.dygraph.save_dygraph(model.state_dict(), model_path)
                fluid.dygraph.save_dygraph(optimizer.state_dict(), model_path)
            f.close()
            evaluate(model, epoch + 1)
    logger.info("Done.")


def main():
    train()


if __name__ == '__main__':
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()