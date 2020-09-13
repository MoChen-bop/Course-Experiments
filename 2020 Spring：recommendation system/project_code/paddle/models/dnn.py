import math
import paddle.fluid as fluid
from utils.config import config as cfg

class DNN(object):

    def __init__(self):
        self.name = 'DNN_'
        for d in cfg.dnn_hidden_dims:
            self.name += '_' +  str(d)

    def input_data(self):
        dense_input = fluid.data(name='dense_input',
                                 shape=[-1, cfg.dense_feature_dim],
                                 dtype='float32')
        
        sparse_input_ids = [
            fluid.data(name="C" + str(i),
                       shape=[-1, 1],
                       lod_level=1,
                       dtype='int64') for i in range(1, 27)]
        label = fluid.data(name='label', shape=[-1, 1], dtype='int64')

        inputs = [dense_input] + sparse_input_ids + [label]
        return inputs
    

    def net(self, inputs):
        def embedding_layer(inputs):
            return fluid.layers.embedding(
                input=inputs,
                is_sparse=True,
                size=[cfg.sparse_feature_dim, cfg.embedding_size],
                param_attr=fluid.ParamAttr(
                    name='SparseFeatFactors',
                    initializer=fluid.initializer.Uniform())
            )
        
        sparse_embed_seq = list(map(embedding_layer, inputs[1:-1]))
        
        feature = fluid.layers.concat(sparse_embed_seq + inputs[0:1], axis=1)

        for hidden_dim in cfg.dnn_hidden_dims:
            feature = fluid.layers.fc(
                input=feature,
                size=hidden_dim,
                act='relu',
                param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                    scale=1/math.sqrt(feature.shape[1])))
            )
        
        predict = fluid.layers.fc(
            input=feature,
            size=2,
            act='softmax',
            param_attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(
                scale=1/math.sqrt(feature.shape[1])))
        )

        cost = fluid.layers.cross_entropy(input=predict, label=inputs[-1])
        avg_cost = fluid.layers.reduce_sum(cost)
        auc_var, _, _ = fluid.layers.auc(input=predict,
                                        label=inputs[-1],
                                        num_thresholds=2**12,
                                        slide_steps=20)
        
        return avg_cost, auc_var
