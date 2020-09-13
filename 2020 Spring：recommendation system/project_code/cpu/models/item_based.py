import os
import sys
sys.path.append('..')
import numpy as np

from models.recommender import Recommender 
from utils.similarity import consine_similarity, pearson_coefficient


class ItemBasedRecommender(Recommender):

    def __init__(self, dataset, ignore_zero_rating=False, exp_name='test', save_path='../saved_models/item_based'):
        if ignore_zero_rating:
            self.name = '__ItemBasedRecommender'
        else:
            self.name = 'ItemBasedRecommender'
        self.dataset = dataset.train_dataset
        self.dataset_full = dataset
        self.save_path = save_path
        self.exp_name = exp_name
        self.ignore_zero_rating = ignore_zero_rating
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model_file = os.path.join(save_path, exp_name + '_default.model')
        self.model = None


    def build_model(self, similarity='cosine', K=20, rebuild=False):
        print('Building model.')
        self.name += '_' + str(K) + '_' + similarity
        self.model_file = os.path.join(self.save_path, 
                self.exp_name + '_%s_%d_%s_default.model' % (self.dataset.name, K, similarity))
        if not rebuild and os.path.exists(self.model_file):
            self.model = super().load_external_model(self.model_file)
            if self.model['measure'] == similarity and self.model['K'] == K:
        	    return

        self.model = {'measure' : similarity, 'K' : K}
        self.model['item_similarities'] = self.__calculate_similarities(similarity, K)
        print('Done.')
        self.save_model()


    def save_model(self):
        print('Saving model to the path: ' + self.model_file)
        super().dump_model(self.model, self.model_file)
        print('Done')


    def recommend(self, user_id, N=20):
        item_ratings = self.__predict_ratings(user_id)[:N]
        return [item[0] for item in item_ratings]


    def recommend_2(self, user_id, N=20):
        item_ratings = self.__predict_ratings_2(user_id)[:N]
        return [item[0] for item in item_ratings]


    def predict(self, user_id, item_id):
        return self.__calculate_rating(user_id, item_id)


    def __calculate_similarities(self, similarity, K):
        if similarity == 'cosine':
            similarity_measure = consine_similarity
        elif similarity == 'pearson':
            similarity_measure = pearson_coefficient

        item_sims = {}
        count = 0
        for item_id1 in self.dataset.item_ids:
            if count % 100 == 0:
                print('Progress: %.2f%%' % (count / len(self.dataset.item_ids) * 100))
            sims = []
            rating1 = self.dataset.item_rates(item_id1)
            for item_id2 in self.dataset.item_ids:
                if item_id2 == item_id1:
                	continue
                rating2 = self.dataset.item_rates(item_id2)
                sim = similarity_measure(rating1, rating2)
                sims.append((item_id2, sim))
            sims.sort(key=lambda k: k[1], reverse=True)
            item_sims[item_id1] = sims[: K]
            count += 1
        return item_sims


    def __predict_ratings(self, user_id):
        item_ratings = []
        for item_id in self.dataset.item_ids:
            item_rating = self.__calculate_rating(user_id, item_id)
            item_ratings.append((item_id, item_rating))
        item_ratings.sort(key=lambda k : k[1], reverse=True)
        return item_ratings


    def __predict_ratings_2(self, user_id):
        item_ratings = []
        for item_id in self.dataset.item_ids:
            rating = self.predict(user_id, item_id)
            if self.dataset_full.train_dataset.rate(user_id, item_id) > 0 or \
                self.dataset_full.test_dataset.rate(user_id, item_id) > 0:
                item_ratings.append((item_id, rating))
        item_ratings.sort(key=lambda k : k[1], reverse=True)
        return item_ratings


    def __calculate_rating(self, user_id, item_id):
        item_sims = self.model['item_similarities'][item_id]
        item_ratings = self.dataset.user_rates(user_id)
        rating = 0
        norm = 0
        for item_id, item_sim in item_sims:
            try:
                item_rating = item_ratings[item_id]
            except:
                item_rating = 0

            if self.ignore_zero_rating and item_rating == 0:
                continue
            rating += item_rating * item_sim
            norm += item_sim
        if rating != 0:
            rating = rating / norm
        else:
            rating = self.dataset.item_mean_rates(item_id)
        return rating


if __name__ == '__main__':
    from datasets.dataset import DataType
    from datasets.movieLens import MovieLensDataset100K, MovieLensDataset1M, MovieLensDataset20M
    dataset = MovieLensDataset100K(type=DataType.ndarray)
    dataset.load_dataset()
    recommender = ItemBasedRecommender(dataset.train_dataset)
    recommender.build_model()
    
    user_ids = dataset.train_dataset.user_ids
    item_ids = dataset.train_dataset.item_ids
    print(recommender.recommend(user_ids[0]))

    print(recommender.predict(user_ids[0], item_ids[0]))

    recommender.save_model()