import math

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Embedding
from utils.config import config as cfg


class DeepFM(fluid.dygraph.Layer):
    def __init__(self):
        super(DeepFM, self).__init__()
        self.name = 'DeepFM_' + str(cfg.embedding_size) + '_' + str(cfg.deepfm_layer_sizes[0])
        self.init_value_ = 0.1

        self.fm = FM()
        self.dnn = DNN()

    def forward(self, raw_feat_idx, raw_feat_value, label):
        feat_idx = fluid.layers.reshape(raw_feat_idx,
                                        [-1, 1])  # (None * num_field) * 1
        feat_value = fluid.layers.reshape(
            raw_feat_value,
            [-1, cfg.num_field, 1])  # None * num_field * 1

        y_first_order, y_second_order, feat_embeddings = self.fm(feat_idx,
                                                                 feat_value)
        y_dnn = self.dnn(feat_embeddings)

        predict = fluid.layers.sigmoid(y_first_order + y_second_order + y_dnn)

        return predict


class FM(fluid.dygraph.Layer):
    def __init__(self):
        super(FM, self).__init__()
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

    def forward(self, feat_idx, feat_value):
        # -------------------- first order term  --------------------
        first_weights_re = self.embedding_w(feat_idx)
        first_weights = fluid.layers.reshape(
            first_weights_re,
            shape=[-1, cfg.num_field, 1])  # None * num_field * 1
        y_first_order = fluid.layers.reduce_sum(first_weights * feat_value, 1)

        # -------------------- second order term  --------------------
        feat_embeddings_re = self.embedding(feat_idx)
        feat_embeddings = fluid.layers.reshape(
            feat_embeddings_re,
            shape=[-1, cfg.num_field, cfg.embedding_size
                   ])  # None * num_field * embedding_size
        feat_embeddings = feat_embeddings * feat_value  # None * num_field * embedding_size

        # sum_square part
        summed_features_emb = fluid.layers.reduce_sum(
            feat_embeddings, 1)  # None * embedding_size
        summed_features_emb_square = fluid.layers.square(
            summed_features_emb)  # None * embedding_size

        # square_sum part
        squared_features_emb = fluid.layers.square(
            feat_embeddings)  # None * num_field * embedding_size
        squared_sum_features_emb = fluid.layers.reduce_sum(
            squared_features_emb, 1)  # None * embedding_size

        y_second_order = 0.5 * fluid.layers.reduce_sum(
            summed_features_emb_square - squared_sum_features_emb,
            1,
            keep_dim=True)  # None * 1

        return y_first_order, y_second_order, feat_embeddings


class DNN(fluid.dygraph.Layer):
    def __init__(self):
        super(DNN, self).__init__()
        self.init_value_ = 0.1
        sizes = [cfg.num_field * cfg.embedding_size] + cfg.deepfm_layer_sizes + [1]
        acts = ['relu'
                for _ in range(len(cfg.deepfm_layer_sizes))] + [None]
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

    def forward(self, feat_embeddings):
        y_dnn = fluid.layers.reshape(
            feat_embeddings,
            [-1, cfg.num_field * cfg.embedding_size])
        for linear in self.linears:
            y_dnn = linear(y_dnn)
        return y_dnn
