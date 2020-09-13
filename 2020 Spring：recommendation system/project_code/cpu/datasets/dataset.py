import abc
import numpy as np
import pandas as pd
from enum import Enum

class DataType(Enum):
    ndarray = "NDArray"
    dataframe = "DataFrame"
    dictionary = "Dictionary"


class Dataset(object):
    def __init__(self, arg):
        super(Dataset, self).__init__()
        self.arg = arg


    @classmethod
    @abc.abstractmethod
    def load_dataset(cls):
    	raise NotImplementedError


    @classmethod
    @abc.abstractmethod	
    def rate(cls):
    	raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def user_rates(cls):
    	raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def item_rates(cls):
    	raise NotImplementedError


class RatingNDArray(Dataset):

    def __init__(self, name, user_num, item_num):
        self.name = name
        self.user_num = user_num
        self.item_num = item_num
        self.user_ids = list(range(1, user_num + 1))
        self.item_ids = list(range(1, item_num + 1))
        self.ratings = np.zeros([len(self.user_ids) + 1, len(self.item_ids) + 1])
        self.time_stamps = np.zeros([self.user_num + 1, self.item_num + 1])

        self.__iter_user_index = 1
        self.__iter_item_index = 1


    def __iter__(self):
        self.__iter_user_index = 1
        self.__iter_item_index = 1
        return self


    def __next__(self):
        begin = True
        user_index = self.__iter_user_index
        while user_index <= self.user_num:
            item_index = 0
            if begin:
                item_index = self.__iter_item_index
                begin = False
            while item_index <= self.item_num:
                if self.ratings[user_index][item_index] != 0:
                    rating = self.ratings[user_index][item_index]
                    if item_index == self.item_num:
                        self.__iter_item_index = 0
                        self.__iter_user_index = user_index + 1
                    else:
                        self.__iter_item_index = item_index + 1
                        self.__iter_user_index = user_index

                    return user_index, item_index, rating
                item_index += 1
            user_index += 1
        raise StopIteration()


    def load_dataset(self, data_frame, time_stamp=True):
        for index, row in data_frame.iterrows():
            if time_stamp:
                user_id, item_id, rate, time_stamp = row
            else:
                user_id, item_id, rate = row
            user_id = int(user_id)
            item_id = int(item_id)
            self.ratings[user_id][item_id] = rate
            if time_stamp:
                self.time_stamps[user_id][item_id] = time_stamp


    def rate(self, user_id, item_id, contain_time=False):
        if contain_time:
            return self.ratings[user_id][item_id], self.time_stamps[user_id][item_id]
        else:
            return self.ratings[user_id][item_id]


    def user_rates(self, user_id, contain_time=False):
        if contain_time:
            return self.ratings[user_id, :], self.time_stamps[user_id, :]
        else:
            return self.ratings[user_id, :]


    def item_rates(self, item_id, contain_time=False):
        if contain_time:
            return self.ratings[:, item_id], self.time_stamps[:, item_id]
        else:
            return self.ratings[:, item_id]


    def user_mean_rates(self, user_id):
        user_ratings = self.user_rates(user_id)
        return user_ratings[user_ratings != 0].mean()


    def item_mean_rates(self, item_id):
        item_ratings = self.item_rates(item_id)
        return item_ratings[item_ratings != 0].mean()


class RatingDataFrame(Dataset):
    
    def __init__(self, name, user_ids, item_ids):
        self.name = name
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = None
        self.item_to_ratings = None

        self.dataframe = None
        self.__iter_index = 0


    def __iter__(self):
        self.__iter_index = 0
        return self


    def __next__(self):
        if self.__iter_index >= len(self.ratings):
            raise StopIteration()
        self.__iter_index += 1
        return self.dataframe.iat[self.__iter_index - 1, 0], \
            self.dataframe.iat[self.__iter_index - 1, 1], self.dataframe.iat[self.__iter_index - 1, 2]


    def load_dataset(self, data_frame, time_stamp=True):
        self.dataframe = data_frame
        self.ratings = data_frame.set_index(['User ID', 'Item ID']).sort_index()
        self.item_to_ratings = data_frame.set_index(['Item ID', 'User ID']).sort_index()


    def rate(self, user_id, item_id, contain_time=False):
        try:
            if contain_time:
                return self.ratings.loc[(user_id, item_id),'Rating'], self.ratings.loc[(user_id, item_id), 'Timestamp']
            else:
                return self.ratings.loc[(user_id, item_id), 'Rating']
        except:
            if contain_time:
                return 0, 0
            else:
                return 0


    def user_rates(self, user_id, contain_time=False):
        try:
            item_ratings = self.ratings.loc[user_id, 'Rating']
            if contain_time:
                rating_timestamps = self.ratings.loc[user_id, 'Timestamp']
        except:
            item_ratings = pd.Series(np.zeros(len(self.item_ids)), index=self.item_ids)
            if contain_time:
                rating_timestamps = pd.Series(np.zeros(len(item_ids)), index=item_ids)

        if contain_time:
            return item_ratings, rating_timestamps
        else:
            return item_ratings


    def item_rates(self, item_id, contain_time=False):
        try:
            user_ratings = self.item_to_ratings.loc[item_id, 'Rating']
            if contain_time:
                user_timestamps = self.item_to_timestamps.loc[item_id, 'Timestamp']
        except:
            user_ratings = pd.Series(np.zeros(len(self.user_ids)), index=self.user_ids)
            if contain_time:
                rating_timestamps = pd.Series(np.zeros(len(user_ids)), index=user_ids)

        if contain_time:
            return user_ratings, rating_timestamps
        else:
            return user_ratings


    def user_mean_rates(self, user_id):
        item_ratings = self.ratings.loc[user_id, 'Rating']
        return item_ratings.mean()


    def item_mean_rates(self, item_id):
        user_ratings = self.item_to_ratings.loc[item_id, 'Rating']
        return user_ratings.mean()


class RatingDictionary(Dataset):

    def __init__(self, name, user_ids, item_ids):
        self.name = name
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.ratings = {}
        self.item_to_ratings = {}

        self.user_to_timestamps = {}
        self.item_to_timestamps = {}

        self.dataframe = None
        self.__iter_index = 0


    def __iter__(self):
        self.__iter_index = 0
        return self


    def __next__(self):
        if self.__iter_index >= len(self.dataframe):
            raise StopIteration()
        self.__iter_index += 1
        return self.dataframe.iat[self.__iter_index - 1, 0], \
            self.dataframe.iat[self.__iter_index - 1, 1], self.dataframe.iat[self.__iter_index - 1, 2]


    def load_dataset(self, data_frame, use_time_stamp=True):
        self.dataframe = data_frame
        for index, row in data_frame.iterrows():
            if use_time_stamp:
                user_id, item_id, rate, time_stamp = row
            else:
                user_id, item_id, rate = row

            #user_id = int(user_id)
            #item_id = int(item_id)
            self.ratings.setdefault(user_id, {})
            self.item_to_ratings.setdefault(item_id, {})
            self.ratings[user_id][item_id] = rate
            self.item_to_ratings[item_id][user_id] = rate

            if use_time_stamp:
                self.user_to_timestamps.setdefault(user_id, {})
                self.item_to_timestamps.setdefault(item_id, {})
                self.user_to_timestamps[user_id][item_id] = time_stamp
                self.item_to_timestamps[item_id][user_id] = time_stamp


    def rate(self, user_id, item_id, contain_time=False):
        try:
            if contain_time:
                return self.ratings[user_id][item_id], self.user_to_timestamps[user_id][item_id]
            else:
                return self.ratings[user_id][item_id]
        except:
            if contain_time:
                return 0, 0
            else:
                return 0


    def user_rates(self, user_id, contain_time=False):
        try:
            item_ratings = self.ratings[user_id].copy()
        except:
            item_ratings = {}
            for item_id in self.item_ids:
                item_ratings[item_id] = 0
        if contain_time:
            try:
                item_timestamps = self.user_to_timestamps[user_id]
            except:
                item_timestamps = {}
                for item_id in self.item_ids:
                    item_timestamps[item_id] = 0

        if contain_time:
            return item_ratings, item_timestamps
        else:
            return item_ratings


    def item_rates(self, item_id, contain_time=False):
        try:
            user_ratings = self.item_to_ratings[item_id].copy()
        except:
            user_ratings = {}
            for user_id in self.user_ids:
                user_ratings[user_id] = 0
        if contain_time:
            try:
                user_timestamps = self.item_to_timestamps[item_id]
            except:
                user_timestamps = {}
                for user_id in self.user.ids:
                    user_timestamps[user_id] = 0

        for user_id in self.user_ids:
            if user_id not in user_ratings:
                user_ratings[user_id] = 0
                if contain_time:
                    user_timestamps[user_id] = 0

        if contain_time:
            return user_ratings, user_timestamps
        else:
            return user_ratings


    def user_mean_rates(self, user_id):
        item_ratings = self.ratings[user_id]
        mean_rating = 0
        for item_id, ratings in item_ratings.items():
            mean_rating += ratings
        mean_rating = mean_rating / len(item_ratings)
        return mean_rating


    def item_mean_rates(self, item_id):
        user_ratings = self.item_to_ratings[item_id]
        mean_rating = 0
        for user_id, ratings in user_ratings.items():
            mean_rating += ratings
        mean_rating = mean_rating / len(user_ratings)
        return mean_rating