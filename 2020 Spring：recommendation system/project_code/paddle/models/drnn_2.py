import math

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Embedding
from utils.config import config as cfg


class DRNN(fluid.dygraph.Layer):
    def __init__(self):
        super(DRNN, self).__init__()
        self.name = 'DRNN_' + str(cfg.embedding_size) + '_' + str(cfg.drnn_hidden_dim) + '_' + str(cfg.drnn_hidden_layer)
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

        self.hidden_1 = Linear(
                cfg.num_field * cfg.embedding_size,
                cfg.drnn_hidden_dim,
                act='relu',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_ / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_)))
        self.add_sublayer('hidden_1', self.hidden_1)
        
        self.hidden_2 = Linear(
                cfg.drnn_hidden_dim,
                cfg.drnn_hidden_dim,
                act='relu',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_ / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_)))
        self.add_sublayer('hidden_2', self.hidden_2)
        
        self.hidden_3 = Linear(
                cfg.drnn_hidden_dim,
                1,
                act='sigmoid',
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_ / math.sqrt(float(10)))),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=self.init_value_)))
        self.add_sublayer('hidden_3', self.hidden_3)

    def forward(self, raw_feat_idx, raw_feat_value, label):
        feat_idx = fluid.layers.reshape(raw_feat_idx, [-1, 1])  # (None * num_field) * 1
        feat_value = fluid.layers.reshape(raw_feat_value, [-1, cfg.num_field, 1])  # None * num_field * 1

        feat_embeddings_re = self.embedding(feat_idx)
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, cfg.num_field, cfg.embedding_size
                   ])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size
        
        feature = fluid.layers.reshape(feat_embeddings, [-1, cfg.num_field * cfg.embedding_size])
        hidden = self.hidden_1(feature)
        for i in range(cfg.drnn_hidden_layer):
            hidden = self.hidden_2(hidden)
        predict = self.hidden_3(hidden)

        #predict = fluid.layers.sigmoid(y_dnn)

        return predict
