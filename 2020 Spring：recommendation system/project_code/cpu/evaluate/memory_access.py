
from __future__ import print_function
import os
import time
import sys
sys.path.append('..')
import numpy as np
import pandas as pd 
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

from datasets.dataset import DataType
from datasets.movieLens import MovieLensDataset100K, MovieLensDataset1M, MovieLensDataset10M
from datasets.movieLens import MovieLensDataset20M, MovieLensDataset25M


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def test_memory_access_speed(dataset, save_dir='../results'):
    print('Begin testing load time...')
    load_time = time.time()
    dataset.load_dataset()
    load_time = time.time() - load_time

    N_row = 1000
    N_col = 1000
    __N_row = 100
    __N_col = 100
    user_ids = dataset.user_ids
    item_ids = dataset.item_ids

    print('Begin testing dataset size...')
    if isinstance(dataset.train_dataset.ratings, dict):
        size = total_size(dataset.train_dataset.ratings)
    elif isinstance(dataset.train_dataset.ratings, pd.DataFrame):
        size = dataset.train_dataset.ratings.memory_usage().sum()
    else:
        size = sys.getsizeof(dataset.train_dataset.ratings)

    print('Begin testing access elements time...')
    avg_access_elements_time = time.time()
    for user_id in user_ids[:__N_row]:
        for item_id in item_ids[:__N_col]:
            rating = dataset.train_dataset.rate(user_id, item_id) + 1
    avg_access_elements_time = time.time() - avg_access_elements_time
    avg_access_elements_time = avg_access_elements_time / (__N_row * __N_col)

    print('Begin testing access row time...')
    avg_access_rows_time = time.time()
    for user_id in user_ids[:N_row]:
        try:
            item_rating = dataset.train_dataset.user_rates(user_id)[item_ids[0]] + 1
        except:
            item_rating = 1
    avg_access_rows_time = time.time() - avg_access_rows_time
    avg_access_rows_time = avg_access_rows_time / N_row

    print('Begin testing access column time...')
    avg_access_columns_time = time.time()
    for item_id in item_ids[:N_col]:
        try:
            user_rating = dataset.train_dataset.item_rates(item_id)[user_ids[0]] + 1
        except:
            user_rating = 1
    avg_access_columns_time = time.time() - avg_access_columns_time
    avg_access_columns_time = avg_access_columns_time / N_col

    print()
    print('Dataset size: %dMB' % (size / 1024 / 1024))
    print('Load dataset time: %.8fs' % load_time)
    print('Access element time: %.8fms (per element)' % (avg_access_elements_time * 1000))
    print('Access row time: %.8fms (per row)' % (avg_access_rows_time * 1000))
    print('Access column time: %.8fms (per column)' % (avg_access_columns_time * 1000))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, dataset.train_dataset.name + '_evaluate_memory_access.txt')
    f = open(save_path, 'w')
    f.write('Dataset size: %dMB\n' % (size / 1024 / 1024))
    f.write('Load dataset time: %.8fs \n' % load_time)
    f.write('Access element time: %.8fms (per element)\n' % (avg_access_elements_time * 1000))
    f.write('Access row time: %.8fms (per row)\n' % (avg_access_rows_time * 1000))
    f.write('Access column time: %.8fms (per column)\n\n' % (avg_access_columns_time * 1000))

    f.close()


def main():
    #print('#1. test datatype: ' + DataType.ndarray.name)
    #dataset = MovieLensDataset100K(type=DataType.ndarray, split_ratio=1)
    #test_memory_access_speed(dataset)
    #print('#2. test datatype: ' + DataType.dataframe.name)
    #dataset = MovieLensDataset25M(type=DataType.dataframe, split_ratio=1)
    #test_memory_access_speed(dataset)
    print('#3. test datatype: ' + DataType.dictionary.name)
    dataset = MovieLensDataset25M(type=DataType.dictionary, split_ratio=1)
    test_memory_access_speed(dataset)

if __name__ == '__main__':
    main()