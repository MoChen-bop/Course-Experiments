import os
import numpy as np 
import pandas as pd

from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioBasicIO as aIO

def extract_features(data_dir):
    m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05
    fs = []
    filess = []
    fns = []
    for root, dirs, files in os.walk(data_dir):
        for d in dirs:
            print("PATH: " + os.path.join(root, d))
            try:
                f, files, fn = aF.directory_feature_extraction(os.path.join(root, d),
                                                            m_win, m_step, s_win, s_step)
                fs.append(f)
                filess.extend(files)
                fns.append(fn)
            except:
                pass
    fs = np.concatenate(fs, axis=0)
    fn.append('beat')
    fn.append('beat_conf')

    frame = pd.DataFrame(fs,columns=fn)
    frame['audio_path'] = filess

    save_path = "./features.csv"
    frame.to_csv(save_path)


def features(file_path):
    fs, s = aIO.read_audio_file(file_path)
    m_win, m_step, s_win, s_step = 1, 1, 0.1, 0.05
    mid_features, short_features, mid_feature_names = aF.mid_feature_extraction(s, fs, round(fs * m_win), 
                                                                            round(fs * m_step), 
                                                                            round(fs * s_win),
                                                                            round(fs * s_step))
    mid_features = np.transpose(mid_features).mean(axis=0)
    beat, beat_conf = aF.beat_extraction(short_features, s_step)
    mid_features = np.append(mid_features, beat)
    mid_features = np.append(mid_features, beat_conf)
    mid_feature_names.append('beat')
    mid_feature_names.append('beat_conf')
    return mid_features, mid_feature_names


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