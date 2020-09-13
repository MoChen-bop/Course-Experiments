import os
import time
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from datasets.movieLens import MovieLensDataset100K
from datasets.movieLens import MovieLensDataset1M

from models.item_based import ItemBasedRecommender
from models.user_based import UserBasedRecommender

def evaluate(model, dataset, print_detail=False):
    print('Begin evaluate model: %s in dataset %s' % (model.name, dataset.test_dataset.name))
    RMSE_test = 0
    train_ratings = dataset.train_dataset.ratings
    test_ratings = dataset.test_dataset.ratings
    count = 0
    if isinstance(test_ratings, np.ndarray):
	    for user_id in dataset.test_dataset.user_ids:
	        for item_id in dataset.test_dataset.item_ids:
	            gold_rating = dataset.test_dataset.rate(user_id, item_id)
	            if gold_rating != 0:
	                rating = model.predict(user_id, item_id)
	                RMSE_test += (gold_rating - rating) ** 2
	                count += 1
    elif isinstance(test_ratings, pd.DataFrame):
        for (user_id, item_id), (gold_rating, _) in test_ratings.iterrows():
            rating = model.predict(user_id, item_id)
            RMSE_test += (gold_rating - rating) ** 2
            count += 1

    RMSE_test = np.sqrt(RMSE_test / count)
    print('RMSE in testdataset: %.4f\n' % RMSE_test)

    RMSE_train = 0
    count = 0
    if isinstance(train_ratings, np.ndarray):
        for user_id in dataset.train_dataset.user_ids:
            for item_id in dataset.train_dataset.item_ids:
                gold_rating = dataset.train_dataset.rate(user_id, item_id)
                if gold_rating != 0:
                    rating = model.predict(user_id, item_id)
                    RMSE_train += (gold_rating - rating) ** 2
                    count += 1
    elif isinstance(train_ratings, pd.DataFrame):
        for (user_id, item_id), (gold_rating, _) in train_ratings.iterrows():
            rating = model.predict(user_id, item_id)
            RMSE_train += (gold_rating - rating) ** 2
            count += 1

    RMSE_train = np.sqrt(RMSE_train / count)
    print('RMSE in traindataset: %.4f\n' % RMSE_train)
    
    save_dir = '../results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_name = os.path.join(save_dir, 'RMSE_%s_%s.txt' % (model.name, dataset.test_dataset.name))
    f = open(save_file_name, 'w')
    f.write('RMSE in testdataset: %.4f\n' % RMSE_test)
    f.write('RMSE in traindataset: %.4f\n' % RMSE_train)
    f.close()

    total_precises = []
    total_recalls = []
    total_f_scores = []
    total_hit_ratios = []
    N = 100
    
    if isinstance(test_ratings, np.ndarray):
        for user_id in dataset.test_dataset.user_ids[:100]:
            precises = []
            recalls = []
            f_scores = []
            hit_ratios = []
            gold_item_ids = set(np.where(test_ratings[user_id] >= 3)[0])
            gold_item_ids = gold_item_ids.union(set(np.where(train_ratings[user_id] >= 3)[0]))
            if len(gold_item_ids) == 0:
                continue
            recommend_list = model.recommend(user_id, N)

            for i in range(1, N + 1):
                hit = len(set(recommend_list[:i]).intersection(gold_item_ids))
                p = hit / i
                r = hit / len(gold_item_ids)
                if p * r == 0:
                    f = 0
                else:
                    f = 2 * p * r / (p + r)
                h_r = hit / len(gold_item_ids)
                precises.append(p)
                recalls.append(r)
                f_scores.append(f)
                hit_ratios.append(h_r)
            total_precises.append(precises)
            total_recalls.append(recalls)
            total_f_scores.append(f_scores)
            total_hit_ratios.append(hit_ratios)
    elif isinstance(test_ratings, pd.DataFrame):
        for (user_id, item_id), (gold_rating, _) in test_ratings.iterrows():
            pass # not implemented
    
    total_precises = np.sum(np.array(total_precises), axis=0) / len(total_precises)
    total_recalls = np.sum(np.array(total_recalls), axis=0) / len(total_recalls)
    total_f_scores = np.sum(np.array(total_f_scores), axis=0) / len(total_f_scores)
    total_hit_ratios = np.sum(np.array(total_hit_ratios), axis=0) / len(total_hit_ratios)

    print('Precise: ')
    print(total_precises[:5])
    print('Recalls: ')
    print(total_recalls[:5])
    print('F-score: ')
    print(total_f_scores[:5])
    print('Hit ratio: ')
    print(hit_ratios[:5])

    result_dict = {'Precises': total_precises, 'Recalls': total_recalls, 
                   'F-scores': total_f_scores, 'Hit ratios': total_hit_ratios}
    result_df = pd.DataFrame(result_dict)

    save_file_name = os.path.join(save_dir, 'precise_%s_%s.csv' % (model.name, dataset.test_dataset.name))
    result_df.to_csv(save_file_name)
    



def main():
    dataset = MovieLensDataset100K()
    #dataset = __MovieLensDataset1M()
    dataset.load_dataset()

    for k in range(10, 20, 10):
	    #model = ItemBasedRecommender(dataset.train_dataset)
	    model = UserBasedRecommender(dataset.train_dataset)
	    model.build_model(similarity='pearson', K=k)

	    evaluate(model, dataset, print_detail=True)





if __name__ == '__main__':
    main()