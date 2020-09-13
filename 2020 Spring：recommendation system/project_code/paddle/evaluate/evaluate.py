import logging
import numpy as np
import pickle
import os
import sys
sys.path.append('..')
import paddle
import paddle.fluid as fluid

from datasets.cirteo_dataset_feeder import CriteoDataset
from models.dnn import DNN
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

    startup_program = fluid.framework.Program()
    test_program = fluid.framework.Program()
    model = DNN()
    model_path = os.path.join(cfg.save_path, model.name + "_epoch_" + str(cfg.test_epoch), "checkpoint")
    
    with fluid.framework.program_guard(test_program, startup_program):
        with fluid.unique_name.guard():
            inputs = model.input_data()
            loss, auc_var = model.net(inputs)

            exe = fluid.Executor(place)
            feeder = fluid.DataFeeder(feed_list=inputs, place=place)

            fluid.load(fluid.default_main_program(), model_path, exe)

            auc_states_names = [
                '_generated_var_0', '_generated_var_1', '_generated_var_2',
                '_generated_var_3'
            ]
            for var in auc_states_names:
                set_zero(var, scope=inference_scope, place=place)
            
            run_index = 0
            infer_auc = 0
            L = []
            for batch_id, data_test in enumerate(test_reader()):
                loss_val, auc_val = exe.run(test_program, 
                    feed=feeder.feed(data_test),
                    fetch_list=[loss, auc_var])
                run_index += 1
                infer_auc = auc_val
                L.append(loss_val / cfg.batch_size)
                if batch_id % cfg.log_interval == 0:
                    logger.info("TEST --> batch: {} loss: {} auc: {}".format(
                        batch_id, loss_val / cfg.batch_size, auc_val))
            
            infer_loss = np.mean(L)
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
