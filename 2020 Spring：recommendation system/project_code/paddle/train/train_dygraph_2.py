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

from models.deepfm import DeepFM
from models.dnn_2 import DNN
from models.drnn_2 import DRNN
from models.dnn_plus import DNNPlus
from utils.config import config as cfg
from utils.config import print_config, update_config
from utils.option import BaseOptions
from datasets.criteo_dataset_reader import CriteoDataset, data_reader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

def evaluate(model, epoch):
    place = fluid.CUDAPlace(0) if cfg.use_cuda else fluid.CPUPlace()
    inference_scope = fluid.Scope()
    test_files = [
        os.path.join(cfg.evaluate_file_path, x) for x in os.listdir(cfg.evaluate_file_path)
    ]
    test_reader = data_reader(cfg.batch_size, test_files, cfg.feat_dict, data_type="train")
        
    with fluid.dygraph.guard(place):
        model.eval()
        logger.info('Begin evaluate model.')

        run_index = 0
        infer_auc = 0.0
        L = []
        auc_metric_test = fluid.metrics.Auc("ROC")
        for batch_id, data in enumerate(test_reader()):
            raw_feat_idx, raw_feat_value, label = zip(*data)
            raw_feat_idx = np.array(raw_feat_idx, dtype=np.int64)
            raw_feat_value = np.array(raw_feat_value, dtype=np.float32)
            label = np.array(label, dtype=np.int64)
            raw_feat_idx, raw_feat_value, label = [
                to_variable(i)
                for i in [raw_feat_idx, raw_feat_value, label]
            ]

            predict = model(raw_feat_idx, raw_feat_value, label)

            loss = fluid.layers.log_loss(input=predict, label=fluid.layers.cast(label, dtype="float32"))
            batch_loss = fluid.layers.reduce_sum(loss)
            
            predict_2d = fluid.layers.concat([1 - predict, predict], 1)
            auc_metric_test.update(preds=predict_2d.numpy(), labels=label.numpy())

            run_index += 1
            L.append(batch_loss.numpy().item() / cfg.batch_size)

            if batch_id % cfg.log_interval == 0:
                logger.info("TEST --> batch: {} loss: {} auc: {}".format(
                    batch_id, batch_loss.numpy().item() / cfg.batch_size, auc_metric_test.eval()))

        infer_loss = np.mean(L)
        infer_auc = infer_auc / run_index
        infer_result = {}
        infer_result['loss'] = infer_loss
        infer_result['auc'] = auc_metric_test.eval()
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
    if cfg.train_model == 'deepfm':
        with fluid.dygraph.guard(place):
            model = DeepFM()
    elif cfg.train_model == 'dnnplus':
        with fluid.dygraph.guard(place):
            model = DNNPlus()
    elif cfg.train_model == 'dnn':
        with fluid.dygraph.guard(place):
            model = DNN()
    elif cfg.train_model == 'drnn':
        with fluid.dygraph.guard(place):
            model = DRNN()
    
    with fluid.dygraph.guard(place):
        optimizer = fluid.optimizer.Adam(
                learning_rate=cfg.learning_rate,
                parameter_list=model.parameters(),
                regularization=fluid.regularizer.L2DecayRegularizer(cfg.reg))
        # optimizer = fluid.optimizer.SGD(learning_rate=cfg.learning_rate,
        #                                 parameter_list=model.parameters())
        file_list = [
            os.path.join(cfg.train_files_path, x) for x in os.listdir(cfg.train_files_path)
        ]
        train_reader = data_reader(cfg.batch_size, file_list, cfg.feat_dict, data_type="train")
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
            total_loss = 0.0
            total_auc = 0.0
            count = 0
            auc_metric = fluid.metrics.Auc('ROC')
            
            if not os.path.isdir(os.path.join(cfg.log_dir, model.name)):
                os.makedirs( os.path.join(cfg.log_dir, model.name))
            log_path = os.path.join(cfg.log_dir, model.name, str(epoch + 1) + '_train_result.log')
            f = open(log_path, 'w+')

            model.train()
            for batch_id, data in enumerate(train_reader()):
                raw_feat_idx, raw_feat_value, label = zip(*data)
                
                raw_feat_idx = np.array(raw_feat_idx, dtype=np.int64)
                raw_feat_value = np.array(raw_feat_value, dtype=np.float32)
                label = np.array(label, dtype=np.int64)
                raw_feat_idx, raw_feat_value, label = [
                    to_variable(i)
                    for i in [raw_feat_idx, raw_feat_value, label]
                ]

                predict = model(raw_feat_idx, raw_feat_value, label)

                loss = fluid.layers.log_loss(
                    input=predict, label=fluid.layers.cast(label, dtype="float32"))
                batch_loss = fluid.layers.reduce_sum(loss)

                total_loss += batch_loss.numpy().item()
                batch_loss.backward()
                optimizer.minimize(batch_loss)
                model.clear_gradients()
                
                count += 1
                predict_2d = fluid.layers.concat([1 - predict, predict], 1)
                auc_metric.update(preds=predict_2d.numpy(), labels=label.numpy())

                if (batch_id + 1) % cfg.log_interval == 0:
                    logger.info(
                        "epoch: %d, batch_id: %d, loss: %.6f, auc: %.6f" % (
                            epoch + 1, batch_id + 1, total_loss / count / cfg.batch_size, auc_metric.eval()))
                
                if (batch_id + 1) % cfg.log_interval_2 == 0:
                    f.write('%d,%d,%.4f,%.4f\n' % (epoch + 1, batch_id + 1, total_loss / count / cfg.batch_size, auc_metric.eval()))

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