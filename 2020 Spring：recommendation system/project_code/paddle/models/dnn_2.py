import math

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Embedding
from utils.config import config as cfg


class DNN(fluid.dygraph.Layer):
    def __init__(self):
        super(DNN, self).__init__()
        self.name = 'DNN_' + str(cfg.embedding_size) + '_' + str(cfg.dnn_hidden_dims[0])
        self.init_value_ = 0.1
        self.embedding = Embedding(
            size=[cfg.num_feat + 1, cfg.embedding_size],
            dtype='float32',
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0,
                    scale=self.init_value_ /
                    math.sqrt(float(cfg.embedding_size)))))

        sizes = [cfg.num_field * cfg.embedding_size] + cfg.deepfm_layer_sizes + [1]
        acts = ['relu' for _ in range(len(cfg.deepfm_layer_sizes))] + [None]
        w_scales = [
            self.init_value_ / math.sqrt(float(10))
            for _ in range(len(cfg.deepfm_layer_sizes))
        ] + [self.init_value_]
        self.linears = []
        for i in range(len(cfg.deepfm_layer_sizes) + 1):
            linear = Linear(
                sizes[i],
                sizes[i + 1],
                act=acts[i],
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=w_scales[i])),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_)))
            self.add_sublayer('linear_%d' % i, linear)
            self.linears.append(linear)


    def forward(self, raw_feat_idx, raw_feat_value, label):
        feat_idx = fluid.layers.reshape(raw_feat_idx, [-1, 1])  # (None * num_field) * 1
        feat_value = fluid.layers.reshape(raw_feat_value, [-1, cfg.num_field, 1])  # None * num_field * 1

        feat_embeddings_re = self.embedding(feat_idx)
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, cfg.num_field, cfg.embedding_size
                   ])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size
        
        y_dnn = fluid.layers.reshape(feat_embeddings, [-1, cfg.num_field * cfg.embedding_size])
        for linear in self.linears:
            y_dnn = linear(y_dnn)

        predict = fluid.layers.sigmoid(y_dnn)

        return predict
