import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd 

from datasets.dataset import DataType, RatingNDArray, RatingDataFrame, RatingDictionary


class PersonalityDataset():
    
    def __init__(self, name='PersonalityDataset', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='G:/dataset/RS/Personality-isf2018'):
        self.name = name
        self.type = type
        self.users_info_path = os.path.join(dataset_path, 'personality-data.csv')
        self.rates_file_path = os.path.join(dataset_path, 'ratings.csv')

        self.split_ratio = split_ratio
        self.users = self.__load_user_info(self.users_info_path)


    def load_dataset(self):
        print('loading rating from path: ' + self.rates_file_path)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_path, self.split_ratio)

        self.user_ids = self.data['User ID'].drop_duplicates().tolist()
        self.item_ids = self.data['Item ID'].drop_duplicates().tolist()

        if self.type == DataType.ndarray:
            self.train_dataset = RatingNDArray(self.name + '_' + self.type.name + '_trainset', 
                len(self.user_ids), len(self.item_ids))
            self.test_dataset = RatingNDArray(self.name + '_' + self.type.name + '_testset', 
                len(self.user_ids), len(self.item_ids))
        elif self.type == DataType.dataframe:
            self.train_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_trainset', 
                self.user_ids, self.item_ids)
            self.test_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_testset', 
                self.user_ids, self.item_ids)
        elif self.type == DataType.dictionary:
            self.train_dataset = RatingDictionary(self.name + '_' + self.type.name + '_trainset', 
                self.user_ids, self.item_ids)
            self.test_dataset = RatingDictionary(self.name + '_' + self.type.name + '_testset', 
                self.user_ids, self.item_ids)

        self.train_dataset.load_dataset(train_data_frame)
        self.test_dataset.load_dataset(test_data_frame)
        print('Done.')


    def __load_user_info(self, file_path):
        return pd.read_csv(file_path, sep=',', skiprows=[0], names=['User ID', 'Openness', 'Agreeableness',
            'Emotional Stability', 'Conscientiousness', 'Extraversion', 'Assigned Metric', 'Assigned Condition',
            'Movie_1', 'Predicted_rating_1', 'Movie_2', 'Predicted_rating_2',
            'Movie_3', 'Predicted_rating_3', 'Movie_4', 'Predicted_rating_4',
            'Movie_5', 'Predicted_rating_5', 'Movie_6', 'Predicted_rating_6',
            'Movie_7', 'Predicted_rating_7', 'Movie_8', 'Predicted_rating_8',
            'Movie_9', 'Predicted_rating_9', 'Movie_10', 'Predicted_rating_10',
            'Movie_11', 'Predicted_rating_11', 'Movie_12', 'Predicted_rating_12',
            'Is Personalized', 'Enjoy Watching'], dtype={'User ID' : str})


    def __load_rate(self, file_path, split_ratio):
        self.data = pd.read_csv(file_path, sep=',', skiprows=[0], names=['User ID', 'Item ID', 'Rating', 'Timestamp'],
            dtype={'User ID' : str, 'Item ID' : np.uint32, 'Rating' : np.float16}, )
        train_data = self.data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = self.data[~self.data.index.isin(train_data.index)]
        return train_data, test_data



if __name__ == '__main__':
    dataset = PersonalityDataset(type=DataType.dictionary)
    dataset.load_dataset()
    user_ids = dataset.user_ids
    item_ids = dataset.item_ids

    print(dataset.train_dataset.rate(user_ids[1], item_ids[122]))

    for i in range(1, 21):
        print(dataset.train_dataset.user_rates(user_ids[10])[item_ids[i]])
    print()
    for i in range(1, 21):
        print(dataset.train_dataset.item_rates(item_ids[10])[user_ids[i]])
    print(dataset.train_dataset.user_mean_rates(user_ids[10]))
    print(dataset.train_dataset.item_mean_rates(item_ids[10]))

    count = 0
    for user_id, item_id, rating in dataset.train_dataset:
        count += 1
    print(count)
