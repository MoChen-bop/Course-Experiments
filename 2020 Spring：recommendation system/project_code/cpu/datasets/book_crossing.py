import os
import sys
sys.path.append('..')
import numpy as np 
import pandas as pd 

from datasets.dataset import DataType, RatingNDArray, RatingDataFrame, RatingDictionary


class BookCrossingDataset():

    def __init__(self, name='BookCrossingDataset', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='G:/dataset/RS/Book-crossing/BX-CSV-Dump'):
        self.name = name
        self.type = type
        self.users_info_path = os.path.join(dataset_path, 'BX-Users.csv')
        self.items_info_path = os.path.join(dataset_path, 'BX-Books.csv')
        self.rates_file_path = os.path.join(dataset_path, 'BX-Book-Ratings.csv')

        self.split_ratio = split_ratio
        self.users = self.__load_user_info(self.users_info_path)
        self.items = self.__load_item_info(self.items_info_path)
        self.data = pd.read_csv(self.rates_file_path, sep=';', skiprows=[0], names=['User ID', 'Item ID', 'Rating'],
            dtype={'User ID' : str, 'Item ID' : str, 'Rating' : np.float16}, encoding='latin1')
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


    def load_dataset(self):
        print('loading rating from path: ' + self.rates_file_path)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_path, self.split_ratio)
        self.train_dataset.load_dataset(train_data_frame, time_stamp=False)
        self.test_dataset.load_dataset(test_data_frame, time_stamp=False)
        print('Done.')


    def __load_user_info(self, file_path):
        return pd.read_csv(file_path, sep=';', skiprows=[0], names=['User ID', 'Location', 'Age'],
            dtype={'User ID' : np.uint32}, encoding='latin1')


    def __load_item_info(self, file_path):
    	return pd.read_csv(file_path, sep=';', skiprows=[0], names=['Item ID', 'Book Title', 
    	    'Book Author', 'Year of Publication', 'Publisher', 'Image URL S', 'Image URL M',
    	    'Image URL L'], dtype={'Item ID' : str}, encoding='latin1')


    def __load_rate(self, file_path, split_ratio):
        train_data = self.data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = self.data[~self.data.index.isin(train_data.index)]
        return train_data, test_data


if __name__ == '__main__':
    '''
    dataset = BookCrossingDataset(type=DataType.dataframe)
    dataset.load_dataset()

    user_ids = dataset.user_ids
    item_ids = dataset.item_ids
    print(dataset.train_dataset.rate(user_ids[10], item_ids[10]))
    print(dataset.train_dataset.user_rates(user_ids[10])[:20])
    print(dataset.train_dataset.item_rates(item_ids[10])[:20])
    print(dataset.train_dataset.user_mean_rates(user_ids[10]))
    print(dataset.train_dataset.item_mean_rates(item_ids[10]))

    count = 0
    for item in dataset.train_dataset:
        count += 1
    print(count)
    '''

    dataset_directory = BookCrossingDataset(type=DataType.dictionary)
    dataset_directory.load_dataset()
    user_ids = dataset_directory.user_ids
    item_ids = dataset_directory.item_ids
    
    for i in range(1, 21):
        print(dataset_directory.train_dataset.user_rates(user_ids[10])[item_ids[i]])
    print()
    for i in range(1, 21):
        print(dataset_directory.train_dataset.item_rates(item_ids[10])[user_ids[i]])
    print(dataset_directory.train_dataset.user_mean_rates(user_ids[10]))
    print(dataset_directory.train_dataset.item_mean_rates(item_ids[10]))

    count = 0
    for user_id, item_id, rating in dataset_directory.train_dataset:
        count += 1
    print(count)