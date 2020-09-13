import os
import sys
sys.path.append('..')
import numpy as np 
import pandas as pd 

from datasets.dataset import DataType, RatingNDArray, RatingDataFrame, RatingDictionary

class MovieLensDataset100K():
    
    def __init__(self, name='MovieLensDataset100K', type=DataType.ndarray, 
        split_ratio=0.8, dataset_path='../data/MovieLens/ml-100k'):

        self.name = name
        self.type = type
        self.users_info_path = os.path.join(dataset_path, 'u.user')
        self.items_info_path = os.path.join(dataset_path, 'u.item')
        self.rates_file_path = os.path.join(dataset_path, 'u.data')

        self.split_ratio = split_ratio
        self.users = self.__load_user_info(self.users_info_path)
        self.items = self.__load_item_info(self.items_info_path)

        occupation_group = self.users.groupby(by='Occupation')
        occupation_list = list(occupation_group.groups.keys())
        self.occupation2id = {}
        for i, occupation in enumerate(occupation_list):
            self.occupation2id[occupation] = i
        self.gender2id = {'M': 0, 'F': 1}

        self.user_ids = self.users['ID'].drop_duplicates().tolist()
        self.item_ids = self.items['ID'].drop_duplicates().tolist()

        self.user_num = len(self.user_ids)
        self.item_num = len(self.item_ids)
        self.user_occupation_num = len(self.occupation2id)

        self.users = self.users.set_index('ID').sort_index()
        self.items = self.items.set_index('ID').sort_index()

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
        self.train_dataset.load_dataset(train_data_frame)
        self.test_dataset.load_dataset(test_data_frame)
        print('Done')


    def train(self):
        for user_id, item_id, rating in self.train_dataset:
            user_info = self.users.loc[user_id]
            item_info = self.items.loc[item_id]
            user_age_id = user_info['Age'] // 10
            user_gender_id = self.gender2id[user_info['Gender']]
            user_occupation_id = self.occupation2id[user_info['Occupation']]

            item_category_one_hot = np.array(item_info[4:], dtype=np.float32)
            yield user_id, user_age_id, user_gender_id, user_occupation_id, item_id, item_category_one_hot, rating
    

    def evaluate(self):
        for user_id, item_id, rating in self.test_dataset:
            user_info = self.users.loc[user_id]
            item_info = self.items.loc[item_id]
            user_age_id = user_info['Age'] // 10
            user_gender_id = self.gender2id[user_info['Gender']]
            user_occupation_id = self.occupation2id[user_info['Occupation']]

            item_category_one_hot = np.array(item_info[4:], dtype=np.float32)
            yield user_id, user_age_id, user_gender_id, user_occupation_id, item_id, item_category_one_hot, rating


    def user_info(self, user_id):
        user_info = self.users.loc[user_id]
        user_age_id = user_info['Age'] // 10
        user_gender_id = self.gender2id[user_info['Gender']]
        user_occupation_id = self.occupation2id[user_info['Occupation']]
        return user_id, user_age_id, user_gender_id, user_occupation_id
    

    def item_info(self, item_id):
        item_info = self.items.loc[item_id]
        item_category_one_hot = np.array(item_info[4:], dtype=np.float32)
        return item_id, item_category_one_hot


    def __load_user_info(self, file_path):
        return pd.read_csv(file_path, sep='|', names=['ID', 'Age', 'Gender', 'Occupation', 'Code'],
            dtype={'ID' : np.uint16, 'Age' : np.uint8}, encoding='utf-8')


    def __load_item_info(self, file_path):
        return pd.read_csv(file_path, sep='|', names=['ID', 'Title', 'Release date', 'Video release date', 'URL', 
    		'Unknown', 'Action', 'Adveenture', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
    		'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
    		'War', 'Wstern'], dtype={'ID' : np.uint16, 'video release data' : str}, encoding='latin1')


    def __load_rate(self, file_path, split_ratio):
        data = pd.read_csv(file_path, sep='\t', names=['User ID', 'Item ID', 'Rating', 'Timestamp'])
        train_data = data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = data[~data.index.isin(train_data.index)]
        return train_data, test_data


class MovieLensDataset1M():
    
    def __init__(self, name='MovieLensDataset1M', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='../data/MovieLens/ml-1m'):
        self.name = name
        self.type = type
        self.users_info_path = os.path.join(dataset_path, 'users.dat')
        self.items_info_path = os.path.join(dataset_path, 'movies.dat')
        self.rates_file_path = os.path.join(dataset_path, 'ratings.dat')

        self.split_ratio = split_ratio
        self.users = self.__load_user_info(self.users_info_path)
        self.items = self.__load_item_info(self.items_info_path)
        
        self.gender2id = {'M': 0, 'F': 1}
        self.category2id = {'Action' : 0, 'Adventure': 1, 'Animation': 2, "Children's": 3, 'Comedy': 4, 'Crime': 5,
            'Documentary': 6, 'Drama': 7, 'Fantasy': 8, 'Film-Noir': 9, 'Horror':  10, 'Musical': 11, 'Mystery': 12,
            'Romance': 13, 'Sci-Fi': 14, 'Thriller': 15, 'War': 16, 'Western': 17}

        self.user_ids = self.users['ID'].drop_duplicates().tolist()
        self.item_ids = self.items['ID'].drop_duplicates().tolist()

        self.user_occupation_num = 21

        self.users = self.users.set_index('ID').sort_index()
        self.items = self.items.set_index('ID').sort_index()

        self.user_ids = list(range(1, 6041))
        self.item_ids = list(range(1, 3953))

        self.user_num = len(self.user_ids)
        self.item_num = len(self.item_ids)

        if self.type == DataType.ndarray:
            self.train_dataset = RatingNDArray(self.name + '_' + self.type.name + '_trainset', len(self.user_ids), len(self.item_ids))
            self.test_dataset = RatingNDArray(self.name + '_' + self.type.name + '_testset', len(self.user_ids), len(self.item_ids))
        elif self.type == DataType.dataframe:
            self.train_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)
        elif self.type == DataType.dictionary:
            self.train_dataset = RatingDictionary(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDictionary(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)


    def load_dataset(self):
        print('loading rating from path: ' + self.rates_file_path)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_path, self.split_ratio)
        self.train_dataset.load_dataset(train_data_frame)
        self.test_dataset.load_dataset(test_data_frame)
        print('Done')


    def train(self):
        for user_id, item_id, rating in self.train_dataset:
            user_info = self.users.loc[user_id]
            item_info = self.items.loc[item_id]
            user_age_id = user_info['Age'] // 10
            user_gender_id = self.gender2id[user_info['Gender']]
            user_occupation_id = user_info['Occupation']

            item_category_one_hot = self.category2onehot(item_info['Tag'])
            yield user_id, user_age_id, user_gender_id, user_occupation_id, item_id, item_category_one_hot, rating
    

    def evaluate(self):
        for user_id, item_id, rating in self.test_dataset:
            user_info = self.users.loc[user_id]
            item_info = self.items.loc[item_id]
            user_age_id = user_info['Age'] // 10
            user_gender_id = self.gender2id[user_info['Gender']]
            user_occupation_id = user_info['Occupation']

            item_category_one_hot = self.category2onehot(item_info['Tag'])
            yield user_id, user_age_id, user_gender_id, user_occupation_id, item_id, item_category_one_hot, rating


    def user_info(self, user_id):
        user_info = self.users.loc[user_id]
        user_age_id = user_info['Age'] // 10
        user_gender_id = self.gender2id[user_info['Gender']]
        user_occupation_id = user_info['Occupation']
        return user_id, user_age_id, user_gender_id, user_occupation_id

    
    def item_info(self, item_id):
        item_info = self.items.loc[item_id]
        item_category_one_hot = self.category2onehot(item_info['Tag'])
        return item_id, item_category_one_hot


    def category2onehot(self, category):
        category_list = category.split('|')
        onehot = np.zeros(len(self.category2id) + 1, dtype=np.float32)
        for c in category_list:
            onehot[self.category2id[c]] = 1
        return onehot
        

    def __load_user_info(self, file_path):
        return pd.read_csv(file_path, sep='::', names=['ID', 'Gender', 'Age', 'Occupation', 'Code'],
            dtype={'ID' : np.uint32,  'Age' : np.int64, 'Occupation': np.int64}, encoding='utf-8', engine='python')


    def __load_item_info(self, file_path):
        return pd.read_csv(file_path, sep='::', names=['ID', 'Title', 'Tag'], dtype={'ID' : np.uint32}, 
            encoding='latin1', engine='python')


    def __load_rate(self, file_path, split_ratio):
        data = pd.read_csv(file_path, sep='::', names=['User ID', 'Item ID', 'Rating', 'Timestamp'],
            dtype={'User ID' : np.uint16, 'Item ID' : np.uint16, 'Rating' : np.float16, 'Timestamp' : np.uint32},
            engine='python')
        train_data = data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = data[~data.index.isin(train_data.index)]
        return train_data, test_data


class MovieLensDataset10M():
    
    def __init__(self, name='MovieLensDataset10M', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='../data/MovieLens//ml-10m/ml-10M100K'):
        self.name = name
        self.type = type
        self.items_info_path = os.path.join(dataset_path, 'movies.dat')
        self.rates_file_path = os.path.join(dataset_path, 'ratings.dat')
        self.user_num = 71567
        self.split_ratio = split_ratio
        self.items = self.__load_item_info(self.items_info_path)
        self.user_ids = list(range(1, self.user_num + 1))
        self.item_ids = self.items['ID'].drop_duplicates().tolist()

        if self.type == DataType.ndarray:
            self.train_dataset = RatingNDArray(self.name + '_' + self.type.name + '_trainset', len(self.user_ids), len(self.item_ids))
            self.test_dataset = RatingNDArray(self.name + '_' + self.type.name + '_testset', len(self.user_ids), len(self.item_ids))
        elif self.type == DataType.dataframe:
            self.train_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)
        elif self.type == DataType.dictionary:
            self.train_dataset = RatingDictionary(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDictionary(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)


    def load_dataset(self):
        print('loading rating from path: ' + self.rates_file_path)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_path, self.split_ratio)
        self.train_dataset.load_dataset(train_data_frame)
        self.test_dataset.load_dataset(test_data_frame)
        print('Done')


    def __load_item_info(self, file_path):
        return pd.read_csv(file_path, sep='::', names=['ID', 'Title', 'Tag'], dtype={'ID' : np.uint16},
            encoding='utf-8', engine='python')


    def __load_rate(self, file_path, split_ratio):
        data = pd.read_csv(file_path, sep='::', names=['User ID', 'Item ID', 'Rating', 'Timestamp'],
            dtype={'User ID' : np.uint16, 'Item ID' : np.uint16, 'Rating' : np.float16, 'Timestamp' : np.int16}, engine='python')
        train_data = data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = data[~data.index.isin(train_data.index)]
        return train_data, test_data


class MovieLensDataset20M():
    
    def __init__(self, name='MovieLensDataset20M', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='../data/MovieLens/ml-20m'):
        self.name = name
        self.type = type
        self.items_info_path = os.path.join(dataset_path, 'movies.csv')
        self.rates_file_path = os.path.join(dataset_path, 'ratings.csv')
        self.user_num = 138493
        self.split_ratio = split_ratio
        self.items = self.__load_item_info(self.items_info_path)
        self.user_ids = list(range(1, self.user_num + 1))
        self.item_ids = self.items['ID'].drop_duplicates().tolist()

        if self.type == DataType.ndarray:
            self.train_dataset = RatingNDArray(self.name + '_' + self.type.name + '_trainset', len(self.user_ids), len(self.item_ids))
            self.test_dataset = RatingNDArray(self.name + '_' + self.type.name + '_testset', len(self.user_ids), len(self.item_ids))
        elif self.type == DataType.dataframe:
            self.train_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)
        elif self.type == DataType.dictionary:
            self.train_dataset = RatingDictionary(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDictionary(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)


    def load_dataset(self):
        print('loading rating from path: ' + self.rates_file_path)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_path, self.split_ratio)
        self.train_dataset.load_dataset(train_data_frame)
        self.test_dataset.load_dataset(test_data_frame)
        print('Done')


    def __load_item_info(self, file_path):
        return pd.read_csv(file_path, sep=',', skiprows=[0], names=['ID', 'Title', 'Tag'], dtype={'ID' : np.uint16},
            encoding='utf-8')


    def __load_rate(self, file_path, split_ratio):
        data = pd.read_csv(file_path, sep=',', skiprows=[0], names=['User ID', 'Item ID', 'Rating', 'Timestamp'],
            dtype={'User ID' : np.uint16, 'Item ID' : np.uint16, 'Rating' : np.float16, 'Timestamp' : np.int16})
        train_data = data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = data[~data.index.isin(train_data.index)]
        return train_data, test_data


class MovieLensDataset25M():
    
    def __init__(self, name='MovieLensDataset25M', type=DataType.dictionary,
        split_ratio=0.8, dataset_path='../data/MovieLens/ml-25m'):
        self.name = name
        self.type = type
        self.items_info_path = os.path.join(dataset_path, 'movies.csv')
        self.rates_file_path = os.path.join(dataset_path, 'ratings.csv')
        self.user_num = 162541
        self.split_ratio = split_ratio
        self.items = self.__load_item_info(self.items_info_path)
        self.user_ids = list(range(1, self.user_num + 1))
        self.item_ids = self.items['ID'].drop_duplicates().tolist()

        if self.type == DataType.ndarray:
            self.train_dataset = RatingNDArray(self.name + '_' + self.type.name + '_trainset', len(self.user_ids), len(self.item_ids))
            self.test_dataset = RatingNDArray(self.name + '_' + self.type.name + '_testset', len(self.user_ids), len(self.item_ids))
        elif self.type == DataType.dataframe:
            self.train_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDataFrame(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)
        elif self.type == DataType.dictionary:
            self.train_dataset = RatingDictionary(self.name + '_' + self.type.name + '_trainset', self.user_ids, self.item_ids)
            self.test_dataset = RatingDictionary(self.name + '_' + self.type.name + '_testset', self.user_ids, self.item_ids)


    def load_dataset(self):
        print('loading rating from path: ' + self.rates_file_path)
        train_data_frame, test_data_frame = self.__load_rate(self.rates_file_path, self.split_ratio)
        self.train_dataset.load_dataset(train_data_frame)
        self.test_dataset.load_dataset(test_data_frame)
        print('Done')


    def __load_item_info(self, file_path):
        return pd.read_csv(file_path, sep=',', skiprows=[0], names=['ID', 'Title', 'Tag'], dtype={'ID' : np.uint16},
            encoding='utf-8')


    def __load_rate(self, file_path, split_ratio):
        data = pd.read_csv(file_path, sep=',', skiprows=[0], names=['User ID', 'Item ID', 'Rating', 'Timestamp'],
            dtype={'User ID' : np.uint16, 'Item ID' : np.uint16, 'Rating' : np.float16, 'Timestamp' : np.int16})
        train_data = data.sample(frac=split_ratio, random_state=0, axis=0)
        test_data = data[~data.index.isin(train_data.index)]
        return train_data, test_data


if __name__ == '__main__':

    #dataset = MovieLensDataset100K(type=DataType.ndarray)
    #dataset_directory = MovieLensDataset100K(type=DataType.dictionary)
    #dataset_dataframe = MovieLensDataset100K(type=DataType.dataframe)


    #dataset = MovieLensDataset1M(type=DataType.ndarray)
    dataset = MovieLensDataset1M(type=DataType.dictionary)
    #dataset_dataframe = MovieLensDataset1M(type=DataType.dataframe)

    dataset.load_dataset()
    for user_id, user_age_id, user_gender_id, user_occupation_id, item_id, item_category_ont_hot, rating in dataset.train():
        pass
        #print("%d\t%d\t%d\t%d\t%d\t%f\t" % (user_id, user_age_id, user_gender_id, user_occupation_id, item_id, rating))
        #print(item_category_ont_hot)
        #print()
    '''
    print(dataset.users.head(5))
    print(dataset.items.head(5))
    print(dataset.train_dataset.user_rates(10)[:20])
    print(dataset.train_dataset.item_rates(10)[:20])
    print(dataset.train_dataset.user_mean_rates(10))
    print(dataset.train_dataset.item_mean_rates(10))

    count = 0
    for item in dataset.train_dataset:
        count += 1
    print(count)

    dataset_dataframe.load_dataset()
    print(dataset_dataframe.users.head(5))
    print(dataset_dataframe.items.head(5))
    print(dataset_dataframe.train_dataset.user_rates(10)[:20])
    print(dataset_dataframe.train_dataset.item_rates(10)[:20])
    print(dataset_dataframe.train_dataset.user_mean_rates(10))
    print(dataset_dataframe.train_dataset.item_mean_rates(10))

    count = 0
    for user_id, item_id, rating in dataset_dataframe.train_dataset:
        count += 1
    print(count)

    dataset_directory.load_dataset()
    print(dataset_directory.users.head(5))
    print(dataset_directory.items.head(5))
    print(dataset_directory.train_dataset.user_mean_rates(10))
    print(dataset_directory.train_dataset.item_mean_rates(10))

    count = 0
    for user_id, item_id, rating in dataset_directory.train_dataset:
        count += 1
    print(count)
    '''