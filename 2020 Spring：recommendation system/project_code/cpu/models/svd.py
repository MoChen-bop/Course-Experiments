import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from models.recommender import Recommender 


class SVDRecommender(Recommender):
    
    def __init__(self, dataset, use_global_mean=True, exp_name='test', save_path='../saved_models/svd'):
        self.name = 'SVDRecommender'
        self.dataset = dataset.train_dataset
        self.dataset_full = dataset
        self.save_path = save_path
        self.exp_name = exp_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_file = os.path.join(save_path, exp_name + '_default.model')
        self.model = None


    def build_model(self, dim=20, lamb=0.15, rebuild=False):
        print('Building model.')
        self.name += '_' + str(dim)
        self.model_file = os.path.join(self.save_path, 
            self.exp_name + '_%s_%d.model' % (self.dataset.name, dim))
        if not rebuild and os.path.exists(self.model_file):
            self.model = super().load_external_model(self.model_file)
            if self.model['dim'] == dim:
                return

        self.model = {'dim': dim}
        user_vectors, item_vectors = self.__svd(self.dataset, dim, lamb)
        self.model['user_vectors'] = user_vectors
        self.model['item_vectors'] = item_vectors
        print('Done.')
        self.save_model()


    def save_model(self):
        print('Saving model to the path: ' + self.model_file)
        super().dump_model(self.model, self.model_file)
        print('Done.')


    def recommend(self, user_id, N=20):
        item_ratings = self.__predict_rating(user_id)[:N]
        return [item[0] for item in item_ratings]


    def recommend_2(self, user_id, N=20):
        item_ratings = self.__predict_rating_2(user_id)[:N]
        return [item[0] for item in item_ratings]


    def predict(self, user_id, item_id):
        return self.__calculate_rating(user_id, item_id)


    def __svd(self, dataset, dim, lamb=0.15, gamma=0.005, max_step=20):
        if isinstance(dataset.ratings, np.ndarray):
            return self.__decompose(dataset, dim)
        elif isinstance(dataset.ratings, pd.DataFrame) or isinstance(dataset.ratings, dict):
            return self.__optimize(dataset, dim, lamb, gamma, max_step)


    def __predict_rating(self, user_id):
        item_ratings = []
        for item_id in self.dataset.item_ids:
            rating = self.__calculate_rating(user_id, item_id)
            item_ratings.append((item_id, rating))
        item_ratings.sort(key=lambda k : k[1], reverse=True)
        return item_ratings


    def __predict_rating_2(self, user_id):
        item_ratings = []
        for item_id in self.dataset.item_ids:
            rating = self.predict(user_id, item_id)
            if self.dataset_full.train_dataset.rate(user_id, item_id) > 0 or \
                self.dataset_full.test_dataset.rate(user_id, item_id) > 0:
                item_ratings.append((item_id, rating))
        item_ratings.sort(key=lambda k : k[1], reverse=True)
        return item_ratings


    def __calculate_rating(self, user_id, item_id):
        user_vector = self.model['user_vectors'][user_id]
        item_vector = self.model['item_vectors'][item_id]
        rating = np.sum(user_vector * item_vector)
        return rating


    def __decompose(self, dataset, dim):
        matrix = dataset.ratings
        U, S, V = np.linalg.svd(matrix, full_matrices=False)
        s_matrix = np.diag(S)
        user_vectors = U[:, :dim]
        item_vectors = np.dot(s_matrix, V)[:dim, :].T
        return user_vectors, item_vectors


    def __optimize(self, dataset, dim, lamb, gamma, max_step):
        user_vectors = {}
        item_vectors = {}
        for user_id in dataset.user_ids:
            user_vectors.setdefault(user_id, np.random.normal(size=dim))
        for item_id in dataset.item_ids:
            item_vectors.setdefault(item_id, np.random.normal(size=dim))

        if not os.path.exists('../logs'):
            os.makedirs('../logs')
        f = open('../logs/%s_%s_%s_%s_%s.loss' % (self.name, self.dataset.name, str(lamb), str(gamma), str(dim)), 'w+')
        
        for step in range(max_step):
            avg_loss = 0
            count = 0
            for user_id, item_id, rating in dataset:
                p = user_vectors[user_id]
                q = item_vectors[item_id]
                distance = np.sum(p * q) - rating
                grad_user_vector = 2 * distance * q + 2 * lamb * p
                grad_item_vector = 2 * distance * p + 2 * lamb * q
                user_vectors[user_id] -= gamma * grad_user_vector
                item_vectors[item_id] -= gamma * grad_item_vector
                avg_loss += distance ** 2
                count += 1
            avg_loss = avg_loss / count
            print("(%d/%d) average loss: %.2f" % (step + 1, max_step, avg_loss))
            f.write('%d, %f\n' % (step, avg_loss))
        f.close()
        return user_vectors, item_vectors


if __name__ == '__main__':
    from datasets.dataset import DataType
    from datasets.movieLens import MovieLensDataset100K

    dataset = MovieLensDataset100K(type=DataType.dictionary)
    dataset.load_dataset()
    recommender = SVDRecommender(dataset.train_dataset)
    recommender.build_model()
    
    user_ids = dataset.train_dataset.user_ids
    item_ids = dataset.train_dataset.item_ids
    print(recommender.recommend(user_ids[0]))

    print(recommender.predict(user_ids[0], item_ids[0]))

    recommender.save_model()