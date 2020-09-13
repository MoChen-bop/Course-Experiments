import os
import numpy as np 
import pandas as pd

from visual_features import ImageFeatureExtractor

def extract_features(data_dir):
    extractor = ImageFeatureExtractor()
    features = []
    filess = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            try:
                f, fn = extractor.extract_features(os.path.join(root, file))
                features.append(f)
                filess.append(os.path.join(root, file))
            except:
                pass
    features = np.stack(features, axis=0)
    frame = pd.DataFrame(features, columns=fn)
    frame['image_path'] = filess
    save_path = "./features.csv"
    frame.to_csv(save_path)


def features(file_path):
    extractor = ImageFeatureExtractor()
    f, fn = extractor.extract_features(file_path)
    return f, fn


def get_result_list(file_path):
    if not os.path.exists(file_path):
        return []
    f, _ = features(file_path)
    save_path = "./features.csv"
    frame = pd.read_csv(save_path, index_col=0)
    frame_normallized = (frame[frame.columns[:-1]] - frame[frame.columns[:-1]].mean()) / (frame[frame.columns[:-1]].std()) 
    f_normalized = (f - frame[frame.columns[:-1]].mean()) / frame[frame.columns[:-1]].std()
    similarities = []
    for index, row in frame_normallized.iterrows():
        sim = np.dot(np.array(row), f_normalized) / (np.linalg.norm(np.array(row)) * np.linalg.norm(f_normalized))
        similarities.append((sim, frame.loc[index][-1]))
    sorted_list = sorted(similarities, key=lambda x : x[0], reverse=True)
    return sorted_list