import math

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Linear, Embedding

from utils.config import config as cfg

class DNN(fluid.dygraph.Layer):

    def __init__(self):
        super(DNN, self).__init__()
        self.name = 'DNN_1024_512_256_dygraph'

        self.embeddings = [
            Embedding(
                size=[cfg.sparse_feature_dim, cfg.embedding_size],
                dtype='float32',
                padding_idx=0,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.TruncatedNormalInitializer(
                        loc=0.0, scale=0.1 / math.sqrt(float(cfg.embedding_size)))))
            for _ in range(26)
        ]
       
        feature_size = 13 + 26 * cfg.embedding_size
        self.block_1 = Linear(
            feature_size, 1024, act='relu',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1 / math.sqrt(float(10)))),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1)))
        
        self.block_2_1 = Linear(
            1024, 1024, act='relu',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1 / math.sqrt(float(10)))),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1)))
        
        self.block_2_2 = Linear(
            1024, 512, act='relu',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1 / math.sqrt(float(10)))),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1)))
        
        self.block_2_3 = Linear(
            512, 256, act='relu',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1 / math.sqrt(float(10)))),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1)))
        
        self.block_3 = Linear(
            256, 2, act='softmax',
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1 / math.sqrt(float(10)))),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.TruncatedNormalInitializer(
                    loc=0.0, scale=0.1)))


    def forward(self, dense_feature, sparse_feature, label):
        
        embedding_vectors = [self.embeddings[i](sparse_feature[:, i]) for i in range(26)]
        embedding_vectors = fluid.layers.concat(embedding_vectors, axis=1)
        feature = fluid.layers.concat([embedding_vectors, dense_feature], axis=1) 
        hidden = self.block_1(feature)
        hidden = self.block_2_1(hidden)
        hidden = self.block_2_2(hidden)
        hidden = self.block_2_3(hidden)
        predict = self.block_3(hidden)
        
        cost = fluid.layers.cross_entropy(input=predict, label=label)
        avg_cost = fluid.layers.reduce_sum(cost)
        auc_var, _, _ = fluid.layers.auc(input=predict,
                                        label=label,
                                        num_thresholds=2**12,
                                        slide_steps=20)
        
        return avg_cost, auc_var
