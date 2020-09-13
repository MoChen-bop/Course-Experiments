from easydict import EasyDict 

config = EasyDict()

config.num_field = 39

config.num_feat = 1086460

config.dense_feature_dim = 13

config.sparse_feature_dim = 1000001

config.embedding_size = 1

config.drnn_hidden_dim = 512

config.drnn_hidden_layer = 4

config.dnn_hidden_dims = [256, 256, 256]

config.deepfm_layer_sizes = [256, 256, 256]

config.batch_size = 4096

config.cpu_num = 4

config.train_files_path = '../data/Criteo_2/train_data'

config.evaluate_file_path = '../data/Criteo_2/test_data'

config.feat_dict = '../data/Criteo_2/aid_data/feat_dict_10.pkl2'

config.train_model = 'drnn'

#config.checkpoint = '../saved_models/FCDNN_1024_512_256/FCDNN_1024_512_256_epoch_8'
config.checkpoint = ''

config.dygraph = True

config.use_cuda = True

config.learning_rate = 0.001

config.reg = 1e-4

config.epoches = 7

config.save_interval = 1

config.save_path = '../saved_models'

config.log_dir = '../logs'

config.log_interval = 100

config.log_interval_2 = 1000

config.test_epoch = 4

def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v

def print_config(config):
    print('=' * 20 + 'Options' + '=' * 20)
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=' * 20 + '  End  ' + '=' * 20)    