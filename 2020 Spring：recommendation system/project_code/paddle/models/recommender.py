import abc
import pickle

import numpy as np 


class Recommender(object):

    def __init__(self, args):
        self.args = args


    @classmethod
    @abc.abstractmethod
    def build_model():
        raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def recommend(cls):
        raise NotImplementedError


    @classmethod
    @abc.abstractmethod
    def predict(cls):
        raise NotImplementedError


    def load_external_model(self, dump_path):
        print('Loading external model from path: ' + dump_path)
        try:
            file = open(dump_path, 'rb')
            model = pickle.load(file)
            file.close()
            print("Done.")
            return model
        except:
            print('Failed.')
            return None


    def dump_model(self, model, dump_path):
        try:
            file = open(dump_path, 'wb')
            pickle.dump(model, file)
            file.close()
        except IOError as e:
            print(e)
