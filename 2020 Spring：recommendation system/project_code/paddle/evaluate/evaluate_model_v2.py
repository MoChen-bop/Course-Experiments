import os
import time
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from datasets.dataset import DataType
from datasets.movieLens import MovieLensDataset100K
from datasets.movieLens import MovieLensDataset1M, MovieLensDataset10M

from models.multi_percetion import MultiPercetionRecommender


def evaluate(model, dataset, args):
    print('Begin evaluate model: %s on dataset %s' % (model.name, dataset.test_dataset.name))
    print('Begin testing model build time...')
    model_build_time = time.time()
    model.build_model(**args['model_parameters'])
    model_build_time = time.time() - model_build_time

    model.build_model(**args['model_parameters'])

    threshold = args['threshold']
    total_precises = []
    total_recalls = []
    total_f_scores = []
    total_nDCG = []
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

        t = time.time()
        recommend_list = model.recommend_2(user_id, N)
        t = time.time() - t
        total_recommend_time += t
        glod_train_item_ratings = dataset.train_dataset.user_rates(user_id)
        glod_test_item_ratings = dataset.test_dataset.user_rates(user_id)
        glod_item_ratings = {}
        glod_recommend_list = []
        gold_rating_list = []
        
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

            precises.append(p)
            recalls.append(r)
            f_scores.append(f)
            nDCGs.append(nDCG)

        total_precises.append(precises)
        total_recalls.append(recalls)
        total_f_scores.append(f_scores)
        total_nDCG.append(nDCGs)
    
    avg_precises = np.zeros(N)
    for i in range(N):
        count = 0
        for precise_list in total_precises:
            if len(precise_list) > i:
                avg_precises[i] += precise_list[i]
                count += 1
        if count != 0:
            avg_precises[i] /= count
    
    avg_recalls = np.zeros(N)
    for i in range(N):
        count = 0
        for recall_list in total_recalls:
            if len(recall_list) > i:
                avg_recalls[i] += recall_list[i]
                count += 1
        if count != 0:
            avg_recalls[i] /= count

    avg_f_scores = np.zeros(N)
    for i in range(N):
        count = 0
        for f_list in total_f_scores:
            if len(f_list) > i:
                avg_f_scores[i] += f_list[i]
                count += 1
        if count != 0:
            avg_f_scores[i] /= count


    avg_nDCG_scores = np.zeros(N)
    for i in range(N):
        count = 0
        for nDCG_list in total_nDCG:
            if len(nDCG_list) > i:
                avg_nDCG_scores[i] += nDCG_list[i]
                count += 1
        if count != 0:
            avg_nDCG_scores[i] /= count
    avg_recommend_time = total_recommend_time / user_count

    save_dir = args['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_dict = {'Precises': avg_precises, 'Recalls': avg_recalls, 'F-scores': avg_f_scores, 
                   'nDCGs': avg_nDCG_scores}
    result_df = pd.DataFrame(result_dict)
    save_file_name = os.path.join(save_dir, 'matrics_%s_%s_2.csv' % (model.name, dataset.test_dataset.name))
    result_df.to_csv(save_file_name)


def main():
    #dataset = MovieLensDataset100K(type=DataType.dictionary)
    dataset = MovieLensDataset1M(type=DataType.dictionary)
    dataset.load_dataset()
    
    model = MultiPercetionRecommender(dataset, use_cuda=True)

    args = {}
    args['model_parameters'] = { 'rebuild': False}
    args['N'] = 700
    args['threshold'] = 3
    args['user_count'] = 1000
    args['save_dir'] = '../results'
    evaluate(model, dataset, args)


if __name__ == '__main__':
    main()