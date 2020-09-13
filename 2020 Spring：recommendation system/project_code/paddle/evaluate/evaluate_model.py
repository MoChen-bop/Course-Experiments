import os
import time
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from datasets.dataset import DataType
from datasets.movieLens import MovieLensDataset100K
from datasets.movieLens import MovieLensDataset1M, MovieLensDataset10M

#from models.item_based import ItemBasedRecommender
#from models.user_based import UserBasedRecommender
#from models.svd import SVDRecommender
#from models.bias_svd import BiasSVDRecommender
#from models.svd_plus_user_based import SVDPlusUserBasedRecommender
from models.multi_percetion import MultiPercetionRecommender

def evaluate(model, dataset, args):
    print('Begin evaluate model: %s on dataset %s' % (model.name, dataset.test_dataset.name))
    print('Begin testing model build time...')
    model_build_time = time.time()
    model.build_model(**args['model_parameters'])
    model_build_time = time.time() - model_build_time
    model.build_model(rebuild=False)

    print('Begin testing RMSE, MAE')
    train_RMSE = 0
    train_MAE = 0
    count = 0
    for user_id, item_id, gold_rating in dataset.train_dataset:
        rating = model.predict(user_id, item_id)
        train_RMSE += (rating - gold_rating) ** 2
        train_MAE += np.abs(rating - gold_rating)
        count += 1
    train_RMSE = np.sqrt(train_RMSE / count)
    train_MAE = train_MAE / count

    test_RMSE = 0
    test_MAE = 0
    count = 0
    for user_id, item_id, gold_rating in dataset.test_dataset:
        rating = model.predict(user_id, item_id)
        test_RMSE += (rating - gold_rating) ** 2
        test_MAE += np.abs(rating - gold_rating)
        count += 1
    test_RMSE = np.sqrt(test_RMSE / count)
    test_MAE = test_MAE / count

    print()
    print('RMSE in traindataset: %.4f\n' % train_RMSE)
    print('MAE in traindataset: %.4f\n' % train_MAE)
    print('RMSE in testdataset: %.4f\n' % test_RMSE)
    print('MAE in testdataset: %.4f\n' % test_MAE)
    print()

    threshold = args['threshold']
    total_precises = []
    total_recalls = []
    total_f_scores = []
    total_nDCG = []
    total_freshness = []
    total_recommend_time = 0

    user_count = args['user_count']
    N = args['N']

    print('Begin testing precises, recalls, recommend_time...')
    for index, user_id in enumerate(dataset.user_ids[:user_count]):
        if index % 50 == 0:
            print('Progress: %.2f%%' % (index / user_count * 100))
        precises = []
        recalls = []
        f_scores = []
        nDCGs = []
        freshness = []

        t = time.time()
        recommend_list = model.recommend(user_id, N)
        t = time.time() - t
        total_recommend_time += t
        
        glod_train_item_ratings = dataset.train_dataset.user_rates(user_id)
        glod_test_item_ratings = dataset.test_dataset.user_rates(user_id)
        glod_item_ratings = {}
        glod_recommend_list = []
        gold_rating_list = []
        fresh_item_list = []
        
        for item_id in dataset.item_ids:
            try:
                train_rating = glod_train_item_ratings[item_id]
            except:
                train_rating = 0
            try:
                test_rating = glod_test_item_ratings[item_id]
            except:
            	test_rating = 0

            if train_rating > 0:
                glod_item_ratings[item_id] = train_rating
                gold_rating_list.append(train_rating)
            else:
                glod_item_ratings[item_id] = test_rating
                gold_rating_list.append(test_rating)

            if train_rating >= threshold or test_rating >= threshold:
                glod_recommend_list.append(item_id)

            if train_rating == 0 and test_rating == 0:
                fresh_item_list.append(item_id)
        
        gold_rating_list.sort(reverse=True)
        hit = 0
        fresh = 0
        nDCG = 0
        DCG = 0
        iDCG = 0
        gold_item_count = len(glod_recommend_list)
        for i, item_id in enumerate(recommend_list):
            if item_id in glod_recommend_list:
                hit += 1

            if item_id in fresh_item_list:
                fresh += 1

            try:
                train_rating = glod_train_item_ratings[item_id]
            except:
                train_rating = 0
            try:
                test_rating = glod_test_item_ratings[item_id]
            except:
            	test_rating = 0
            gold_rating = train_rating + test_rating

            DCG += (np.power(2, gold_rating) - 1) / np.log(i + 2)
            iDCG += (np.power(2, gold_rating_list[i]) - 1) / np.log(i + 2)
            nDCG = DCG / iDCG

            p = hit / (i + 1)
            r = hit / gold_item_count
            f = 2 * p * r 
            if f != 0:
                f = f / (p + r)
            _freshness = fresh / (i + 1)

            precises.append(p)
            recalls.append(r)
            f_scores.append(f)
            nDCGs.append(nDCG)
            freshness.append(_freshness)

        total_precises.append(precises)
        total_recalls.append(recalls)
        total_f_scores.append(f_scores)
        total_nDCG.append(nDCGs)
        total_freshness.append(freshness)

    avg_precises = np.sum(np.array(total_precises), axis=0) / user_count
    avg_recalls = np.sum(np.array(total_recalls), axis=0) / user_count
    avg_f_scores = np.sum(np.array(total_f_scores), axis=0) / user_count
    avg_nDCG = np.sum(np.array(total_nDCG), axis=0) / user_count
    avg_freshness = np.sum(np.array(total_freshness), axis=0) / user_count
    avg_recommend_time = total_recommend_time / user_count

    save_dir = args['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_file_name = os.path.join(save_dir, 'RMSE_MAE_%s_%s.txt' % (model.name, dataset.test_dataset.name))
    f = open(save_file_name, 'w')
    f.write('RMSE in traindataset: %.4f\n' % train_RMSE)
    f.write('MAE in traindataset: %.4f\n' % train_MAE)
    f.write('RMSE in testdataset: %.4f\n' % test_RMSE)
    f.write('MAE in traindataset: %.4f\n' % test_MAE)
    f.write('Recommend time: %.4fms\n' % (avg_recommend_time * 1000))
    f.write('Model building time: %.4fs\n' % model_build_time)
    f.close()

    result_dict = {'Precises': avg_precises, 'Recalls': avg_recalls, 'F-scores': avg_f_scores, 
                   'nDCGs': avg_nDCG, 'Freshness': avg_freshness}
    result_df = pd.DataFrame(result_dict)
    save_file_name = os.path.join(save_dir, 'matrics_%s_%s.csv' % (model.name, dataset.test_dataset.name))
    result_df.to_csv(save_file_name)


def main():
    #dataset = MovieLensDataset100K(type=DataType.dictionary)
    dataset = MovieLensDataset1M(type=DataType.dictionary)
    dataset.load_dataset()
    
    model = MultiPercetionRecommender(dataset, use_cuda=True)

    args = {}
    args['model_parameters'] = { 'rebuild': False}
    args['N'] = 500
    args['threshold'] = 3
    args['user_count'] = 900
    args['save_dir'] = '../results'
    evaluate(model, dataset, args)


if __name__ == '__main__':
    main()