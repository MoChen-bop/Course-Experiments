import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd

from models.recommender import Recommender 
from utils.similarity import consine_similarity, pearson_coefficient


class SVDPlusUserBasedRecommender(Recommender):
    
    def __init__(self, dataset, ignore_zero_rating=False, exp_name='test', save_path='../saved_models/svd_plus_user_based'):
        if ignore_zero_rating:
            self.name = 'SVDPlus__UserBasedRecommender'
        else:
            self.name = 'SVDPlusUserBasedRecommender'
        self.ignore_zero_rating = ignore_zero_rating
        self.dataset = dataset.train_dataset
        self.dataset_full = dataset
        self.save_path = save_path
        self.exp_name = exp_name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_file = os.path.join(save_path, exp_name + '_default.model')
        self.model = None


    def build_model(self, dim=20, similarity='cosine', K=20, rebuild=False):
        print('Building model.')
        self.name += '_' + str(dim) + '_' + similarity + '_' + str(K)
        self.model_file = os.path.join(self.save_path, 
            self.exp_name + '_%d.model' % dim)
        if not rebuild and os.path.exists(self.model_file):
            self.model = super().load_external_model(self.model_file)
            if self.model['dim'] == dim and self.model['measure'] == similarity \
                and self.model['K'] == K:
                return

        self.model = {'dim': dim}
        self.model['measure'] = similarity
        self.model['K'] = K
        user_vectors, item_vectors = self.__svd(self.dataset, dim)
        self.model['user_similarities'] = self.__calculate_similarities(similarity, K, user_vectors)
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


    def __calculate_rating(self, user_id, item_id):
        user_sims = self.model['user_similarities'][user_id]
        user_ratings = self.dataset.item_rates(item_id)
        user_mean_ratings = self.dataset.user_mean_rates(user_id)
        rating = 0
        norm = 0
        for user_id, user_sim in user_sims:
            if self.ignore_zero_rating and user_ratings[user_id] == 0:
                continue

            user_mean_ratings = self.dataset.user_mean_rates(user_id)
            rating += (user_ratings[user_id] - user_mean_ratings) * user_sim
            norm += user_sim
        if norm == 0:
            rating = user_mean_ratings
        else:
            rating = rating / norm + user_mean_ratings
        return rating


    def __calculate_similarities(self, similarity, K, user_vectors):
        if similarity == 'cosine':
            similarity_measure = consine_similarity
        elif similarity == 'pearson':
            similarity_measure = pearson_coefficient

        user_sims = {}
        count = 0
        for user_id1 in self.dataset.user_ids:
            sims = []
            if count % 100 == 0:
                print('Progress: %.2f%%' % (count / len(self.dataset.user_ids) * 100))
            vector1 = user_vectors[user_id1]
            for user_id2 in self.dataset.user_ids:
                if user_id2 == user_id1:
                    continue
                vector2 = user_vectors[user_id2]
                sim = similarity_measure(vector1, vector2)
                sims.append((user_id2, sim))
            sims.sort(key=lambda k: k[1], reverse=True)
            user_sims[user_id1] = sims[: K]
            count += 1
        return user_sims


    def __decompose(self, dataset, dim):
        matrix = dataset.ratings
        U, S, V = np.linalg.svd(matrix, full_matrices=False)
        s_matrix = np.diag(S)
        user_vectors = U[:, :dim]
        item_vectors = np.dot(s_matrix, V)[:dim, :]
        return user_vectors, item_vectors


    def __optimize(self, dataset, dim, lamb, gamma, max_step):
        user_vectors = {}
        item_vectors = {}
        for user_id in dataset.user_ids:
            user_vectors.setdefault(user_id, np.random.normal(size=dim))
        for item_id in dataset.item_ids:
            item_vectors.setdefault(item_id, np.random.normal(size=dim))

        if not os.path.exists('../logs'):
            os.path.makedirs('../logs')
        f = open('../logs/%s_%s_%s_%s_%s.loss' % (self.name, self.dataset.name, str(lamb), 
            str(gamma), str(dim)), 'w+')
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

    dataset = MovieLensDataset100K(type=DataType.dataframe)
    dataset.load_dataset()
    recommender = SVDPlusUserBasedRecommender(dataset.train_dataset)
    recommender.build_model()
    
    user_ids = dataset.train_dataset.user_ids
    item_ids = dataset.train_dataset.item_ids
    print(recommender.recommend(user_ids[0]))

    print(recommender.predict(user_ids[0], item_ids[0]))

    recommender.save_model()