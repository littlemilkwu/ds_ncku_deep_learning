import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from config import *


def union_shuffle(X, y):
    index = np.arange(X.shape[0])
    # np shuffle is inplace edit
    RS.shuffle(index)
    return X[index], y[index]

def read_img_feature_data(part='train'):
    data = np.load(f"{DATA_PATH}/../code/output/{part}_features.npz")
    return data['X'], data['y']

def one_hot_encoding(y):
    y_one = np.zeros((y.size, y.max() + 1))
    y_one[np.arange(y.size), y] = 1
    return y_one