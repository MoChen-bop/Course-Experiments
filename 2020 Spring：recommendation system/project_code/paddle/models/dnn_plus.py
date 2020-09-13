import math

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Embedding
from utils.config import config as cfg


class DNNPlus(fluid.dygraph.Layer):
    def __init__(self):
        super(DNNPlus, self).__init__()
        self.name = 'DNNPlus_' + str(cfg.embedding_size) + '_' + str(cfg.dnn_hidden_dims[0])
        self.init_value_ = 0.1
        self.embedding_w = Embedding(
            size=[cfg.num_feat + 1, 1],
            dtype='float32',
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=self.init_value_),
                regularizer=fluid.regularizer.L1DecayRegularizer(
                    cfg.reg)))
        self.embedding = Embedding(
            size=[cfg.num_feat + 1, cfg.embedding_size],
            dtype='float32',
            padding_idx=0,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0,
                    scale=self.init_value_ /
                    math.sqrt(float(cfg.embedding_size)))))

        self.first_order_act = fluid.layers.sigmoid

        sizes = [cfg.num_field * cfg.embedding_size] + cfg.deepfm_layer_sizes + [1]
        acts = ['relu' for _ in range(len(cfg.deepfm_layer_sizes))] + [None]
        w_scales = [
            self.init_value_ / math.sqrt(float(10))
            for _ in range(len(cfg.deepfm_layer_sizes))
        ] + [self.init_value_]
        
        self.second_order_fc = Linear(
                cfg.deepfm_layer_sizes[0],
                1,
                act='sigmoid',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=w_scales[0])),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_)))
        self.add_sublayer('secong_order', self.second_order_fc)

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

        first_weights_re = self.embedding_w(feat_idx)
        first_weights = fluid.layers.reshape(
            first_weights_re,
            shape=[-1, cfg.num_field, 1])  # None * num_field * 1
        y_first_order = self.first_order_act(fluid.layers.reduce_sum(first_weights * feat_value))

        feat_embeddings_re = self.embedding(feat_idx)
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, cfg.num_field, cfg.embedding_size
                   ])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size
        
        features = fluid.layers.reshape(feat_embeddings, [-1, cfg.num_field * cfg.embedding_size])
        y_dnn = self.linears[0](features)
        y_second_order = self.second_order_fc(y_dnn)
        for linear in self.linears[1:]:
            y_dnn = linear(y_dnn)

        predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict
