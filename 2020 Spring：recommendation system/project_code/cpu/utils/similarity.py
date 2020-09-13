import math
import pandas as pd 
import numpy as np


def consine_similarity(vector1, vector2):
    if isinstance(vector1, pd.Series):
        cos = vector1.mul(vector2, fill_value=0).sum()
        norm1 = math.sqrt(vector1.mul(vector1).sum())
        norm2 = math.sqrt(vector2.mul(vector2).sum())
        if cos != 0:
            cos = cos / (norm1 * norm2)
        return cos
    elif isinstance(vector1, np.ndarray):
        cos = np.sum(vector1 * vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if cos != 0:
            # assert if cos != 0, then norm1 != 0 and norm2 != 0
            cos = cos / (norm1 * norm2)
        return cos
    elif isinstance(vector1, dict):
        cos = 0
        norm1 = 0
        for id, rating in vector1.items():
            norm1 += rating ** 2
            if id in vector2:
                cos += rating * vector2[id]
        norm2 = 0
        for id, rating in vector2.items():
            norm2 += rating ** 2
        if cos != 0:
            cos = cos / (math.sqrt(norm1) * math.sqrt(norm2))
        return cos
    else:
        print("Type Error.")
        return -1


def pearson_coefficient(vector1, vector2):
    if isinstance(vector1, pd.Series):
        vector1 = vector1 - vector1.mean()
        vector2 = vector2 - vector2.mean()
        return consine_similarity(vector1, vector2)
    elif isinstance(vector1, np.ndarray):
        vector1 = vector1 - np.mean(vector1, axis=0)
        vector2 = vector2 - np.mean(vector2, axis=0)
        return consine_similarity(vector1, vector2)
    elif isinstance(vector1, dict):
        mean1 = 0
        for id, rating in vector1:
            mean1 += rating
        mean1 = mean1 / len(vector1)
        for id in vector1.keys():
            vector1[id] = vector1[id] - mean1

        mean2 = 0
        for id, rating in vector2:
            mean2 += rating
        mean2 = mean2 / len(vector2)
        for id in vector2.keys():
            vector2[id] = vector2[id] - mean2
        return consine_similarity(vector1, vector2)
    else:
        print("Type Error.")
        return -1