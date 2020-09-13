from easydict import EasyDict

config = EasyDict()

config.LR = 1

config.K = 10

config.dimVectors = 15 # default 10

config.batchsize = 50 # default 50

config.SAVE_PARAMS_EVERY = 10000

config.log_frequency = 100

config.max_iteration = 10000

config.ANNEAL_EVERY = 10000

config.normalization = False

config.log_dir = './logs'

config.save_path = './save'

config.visual_path = './vis'

config.negSample = True

config.exp_name = 'native_lr_1_dim_15'


def update_config(config, extra_config):
	for k, v in vars(extra_config).items():
		config[k] = v
	config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
	print('========== Options ==========')
	for k, v in config.items():
		print('{}: {}'.format(k, v))
	print('==========   End   ==========')
