import os
import sys
sys.path.append('..')
import pandas as pd 
import numpy as np

from datasets.dataset import DataType, RatingNDArray, RatingDataFrame, RatingDictionary


class NetflixDataset():
    
    def __init__(self, name='NetflixDataset', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='G:/dataset/RS/netflix-prize-data'):
        self.name = name
        self.type = type
        self.items_info_path = os.path.join(dataset_path, 'movie_titles.txt')
        self.rates_file_dir = os.path.join(dataset_path, 'training_set')

        self.split_ratio = split_ratio
        self.items = self.__load_item_info(self.items_info_path)


    def load_dataset(self):
        print('loading ratings from path: ' + self.rates_file_dir)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_dir, self.split_ratio)

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
        print ('Done.')


    def __load_item_info(self, file_path):
        return pd.read_csv(file_path, sep=',', names=['ID', 'Year of Release', 'Title'],
            dtype={'ID' : np.uint32}, encoding='latin1')	


    def __load_rate(self, file_dir, split_ratio):
        self.data = pd.DataFrame(columns=['User ID', 'Item ID', 'Rating', 'Timestamp'])

        for file in os.listdir(file_dir):
            user_id = int(file.split('.')[0].split('_')[1])
            if user_id % 100 == 0:
                print('loading file: ' + file)

            rating_data = pd.read_csv(os.path.join(file_dir, file), sep=',', skiprows=[0], encoding='latin1',
                names=['Item ID', 'Rating', 'Timestamp'], dtype={'Item ID' : np.uint32, 'Rating' : np.float16})
            rating_data['User ID'] = np.uint32(user_id)

            self.data = self.data.append(rating_data)

        train_data = self.data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = self.data[~self.data.index.isin(train_data.index)]
        return train_data, test_data



if __name__ == '__main__':
    
    dataset = NetflixDataset(type=DataType.dictionary)
    dataset.load_dataset()
    user_ids = dataset.user_ids
    item_ids = dataset.item_ids

    print(dataset.train_dataset.rate(1, 122))
    print(dataset.train_dataset.user_rates(1)[:20])
    print(dataset.train_dataset.item_rates(1)[:20])

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