import logging
import numpy as np
import pickle
import os
import sys
sys.path.append('..')
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable

from datasets.criteo_dataset_feeder_dygraph import CriteoDataset
from models.drnn import DRNN
from models.fcdnn import FCDNN
from models.dnn_dygraph import DNN
from utils.config import config as cfg
from utils.config import print_config, update_config
from utils.option import BaseOptions

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)

def evaluate():
    place = fluid.CUDAPlace(0) if cfg.use_cuda else fluid.CPUPlace()
    inference_scope = fluid.Scope()
    test_files = [
        os.path.join(cfg.evaluate_file_path, x) for x in os.listdir(cfg.evaluate_file_path)
    ]
    dataset = CriteoDataset()
    test_reader = paddle.batch(dataset.test(test_files), batch_size=cfg.batch_size)

    with fluid.dygraph.guard(place):
        if cfg.train_model == 'drnn':
            model = DRNN()
        elif cfg.train_model == 'dnn':
            model = DNN()
        elif cfg.train_model == 'fcdnn':
            model = FCDNN()
        model_path = os.path.join(cfg.save_path, model.name, model.name + "_epoch_" + str(cfg.test_epoch))
        
        model_dict, optimizer_dict = fluid.dygraph.load_dygraph(model_path)
        model.set_dict(model_dict)
        logger.info("load model {} finished.".format(model_path))

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
        if not os.path.isdir(cfg.log_dir):
            os.makedirs(cfg.log_dir)
        log_path = os.path.join(cfg.log_dir, model.name + '_infer_result.log')
        
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


if __name__ == '__main__':
    evaluate()
