import os
import sys
sys.path.append('..')
import math
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.nets as nets

from models.recommender import Recommender

class MultiPercetionRecommender(Recommender):

    def __init__(self, dataset, exp_name='test', use_cuda=True, save_path='../saved_models/multi_percetion'):
        self.name = "MultiPercetionRecommender" 
        self.dataset = dataset

        self.user_num = self.dataset.user_num
        self.item_num = self.dataset.item_num
        self.user_occupation_num = self.dataset.user_occupation_num

        self.save_path = save_path
        self.exp_name = exp_name
        self.use_cuda = use_cuda
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_file = os.path.join(save_path, exp_name + '_default.model')
        self.model = None
    

    def build_model(self, embedding_size=16, hidden_size=200, rebuild=False):
        print('Building model.')
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.name += '_' + str(embedding_size) + '_' + str(hidden_size)
        self.model_file = os.path.join(self.save_path,
            self.exp_name + '_%s_%d_%d' % (self.dataset.name, embedding_size, hidden_size))

        if not rebuild and os.path.exists(self.model_file):
            print('Loading model from path: ' + self.model_file)
            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            exe = fluid.Executor(place)
            self.exe = exe
            self.place = place
            [program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(self.model_file, self.exe)
            self.model = {}
            self.model['program'] = program
            self.model['feed_target_names'] = feed_target_names
            self.model['fetch_targets'] = fetch_targets
        else:
            self.model = {}
            self.model['feed_target_names'] = ['user_id', 'user_age_id',  
                'user_gender_id', 'user_occupation_id', 'item_id', 'item_category_one_hot']
            
            place = fluid.CUDAPlace(0) if self.use_cuda else fluid.CPUPlace()
            self.place = place
            main_program = fluid.default_main_program()
            star_program = fluid.default_startup_program()

            scale_infer, avg_cost = self.__model_dataflow()
        
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.2)
            sgd_optimizer.minimize(avg_cost)
            exe = fluid.Executor(place)
            self.exe = exe

            self.model['program'] = main_program
            self.model['fetch_targets'] = [scale_infer]

            self.train(main_program, star_program, self.model['feed_target_names'] + ['rating'], [scale_infer.name, avg_cost.name], exe, place)
            self.save_model(self.model['feed_target_names'], self.model['fetch_targets'], main_program, exe)
   

    def train(self, main_program, star_program, feed_order, fetch_list, exe, place):
        train_reader = paddle.batch(self.dataset.train, batch_size=32)
        
        feed_list = [main_program.global_block().var(var_name) for var_name in feed_order]
        feeder = fluid.DataFeeder(feed_list, place)
        exe.run(star_program)
        epoch_num = 20
        for epoch in range(epoch_num):
            for batch_id, data in enumerate(train_reader()):
                infer, cost = exe.run(program=main_program, feed=feeder.feed(data), fetch_list=fetch_list)
            
                if batch_id % 200 == 0:
                    print('EpochID {0}, BatchID {1}, Train Loss {2:0.2}, Glod_rating {3:0.2}, Predict {4:0.2}'.format(
                        epoch + 1, batch_id + 1, float(cost), float(data[0][6]), float(infer[0])))


    def save_model(self, feed_target_names, fetch_targets, main_program, exe):
        print('Saving model to the path: ' + self.model_file)
        fluid.io.save_inference_model(self.model_file, feed_target_names, fetch_targets, exe)


    def recommend(self, user_id, N=20):
        item_ratings = self.__predict_rating(user_id)[:N]
        return [item[0] for item in item_ratings]
    

    def recommend_2(self, user_id, N=20):
        item_ratings = self.__predict_rating_2(user_id)[:N]
        return [item[0] for item in item_ratings]


    def predict(self, user_id, item_id):
        user_id, user_age_id, user_gender_id, user_occupation_id = self.dataset.user_info(user_id)
        try:
            item_id, item_category_one_hot = self.dataset.item_info(item_id)
        except:
            return 0

        user_id = np.int64(user_id)
        item_id = np.int64(item_id)
        user_id = fluid.create_lod_tensor([[user_id]], [[1]], self.place)
        user_gender_id = fluid.create_lod_tensor([[user_gender_id]], [[1]], self.place)
        user_age_id = fluid.create_lod_tensor([[user_age_id]], [[1]], self.place)
        user_occupation_id = fluid.create_lod_tensor([[user_occupation_id]], [[1]], self.place)
        item_id = fluid.create_lod_tensor([[item_id]], [[1]], self.place)
        t = fluid.LoDTensor()
        item_category_one_hot = item_category_one_hot.reshape(1, -1, 1)
        t.set(item_category_one_hot, self.place)
        out = self.exe.run(self.model['program'],
            feed={
                self.model['feed_target_names'][0]: user_id,
                self.model['feed_target_names'][1]: user_age_id,
                self.model['feed_target_names'][2]: user_gender_id,
                self.model['feed_target_names'][3]: user_occupation_id,
                self.model['feed_target_names'][4]: item_id,
                self.model['feed_target_names'][5]: t,
            },
            fetch_list=self.model['fetch_targets'],
            return_numpy=False)
        predict_rating = np.array(out[0])
        return predict_rating[0][0]


    def __predict_rating(self, user_id):
        item_ratings = []
        for item_id in self.dataset.item_ids:
            rating = self.predict(user_id, item_id)
            item_ratings.append((item_id, rating))
        item_ratings.sort(key=lambda k : k[1], reverse=True)
        return item_ratings

    
    def __predict_rating_2(self, user_id):
        item_ratings = []
        for item_id in self.dataset.item_ids:
            rating = self.predict(user_id, item_id)
            if self.dataset.train_dataset.rate(user_id, item_id) > 0 or \
                self.dataset.test_dataset.rate(user_id, item_id) > 0:
                item_ratings.append((item_id, rating))
        item_ratings.sort(key=lambda k : k[1], reverse=True)
        return item_ratings


    def __model_dataflow(self):
        user_id = layers.data(name='user_id', shape=[1], dtype='int64')
        user_gender_id = layers.data(name='user_gender_id', shape=[1], dtype='int64')
        user_age_id = layers.data(name='user_age_id', shape=[1], dtype='int64')
        user_occupation_id = layers.data(name='user_occupation_id', shape=[1], dtype='int64')
        item_id = layers.data(name='item_id', shape=[1], dtype='int64')
        item_category_one_hot = layers.data(name='item_category_one_hot', shape=[19], dtype='float32')

        user_emb = layers.embedding(input=user_id, dtype='float32', size=[self.user_num + 1, 16],
            param_attr='user_table', is_sparse=True)
        user_gender_emb = layers.embedding(input=user_gender_id, dtype='float32', size=[2, 16],
            param_attr='user_gender_table', is_sparse=True)
        user_age_emb = layers.embedding(input=user_age_id, dtype='float32', size=[10, 16],
            param_attr='user_age_table', is_sparse=True)
        user_occupation_emb = layers.embedding(input=user_occupation_id, dtype='float32', size=[self.user_occupation_num, 16],
            param_attr='user_occupation_table', is_sparse=True)
        item_emb = layers.embedding(input=item_id, dtype='float32', size=[self.item_num + 1, 16],
            param_attr='item_table', is_sparse=True)
        item_category_emb = layers.fc(input=item_category_one_hot, size=16)

        user_fc = layers.fc(input=user_emb, size=32)
        user_gender_fc = layers.fc(input=user_gender_emb, size=16)
        user_age_fc = layers.fc(input=user_age_emb, size=16)
        user_occupation_fc = layers.fc(input=user_occupation_emb, size=16)
        item_fc = layers.fc(input=item_emb, size=32)
        item_category_fc = layers.fc(input=item_category_emb, size=16)

        user_concat_embed = layers.concat(
            input=[user_fc, user_gender_fc, user_age_fc, user_occupation_fc], axis=1)
        item_concat_embed = layers.concat(
            input=[item_fc, item_category_fc], axis=1)

        user_conbined_features = layers.fc(input=user_concat_embed, size=200, act="tanh")
        item_conbined_features = layers.fc(input=item_concat_embed, size=200, act="tanh")

        inference = layers.cos_sim(X=user_conbined_features, Y=item_conbined_features)
        scale_infer = layers.scale(x=inference, scale=5.0)

        rating = layers.data(name='rating', shape=[1], dtype='float32')
        square_cost = layers.square_error_cost(input=scale_infer, label=rating)
        avg_cost = layers.mean(square_cost)

        return scale_infer, avg_cost


if __name__ == '__main__':
    from datasets.dataset import DataType
    from datasets.movieLens import MovieLensDataset100K, MovieLensDataset1M

    dataset = MovieLensDataset100K(type=DataType.dictionary)
    #dataset = MovieLensDataset1M(type=DataType.dictionary)
    dataset.load_dataset()
    recommender = MultiPercetionRecommender(dataset, use_cuda=True)
    recommender.build_model(rebuild=True)
    
    user_ids = dataset.train_dataset.user_ids
    item_ids = dataset.train_dataset.item_ids
    print(recommender.recommend(user_ids[0]))

    print(recommender.predict(user_ids[0], item_ids[0]))














