import gzip
import pickle
import random
import cv2
import numpy as np
from urllib import request
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count


from config import *

def union_shuffle(X, y):
    index = np.arange(X.shape[0])
    # np shuffle is inplace edit
    RS.shuffle(index)
    return X[index], y[index]

def img2np(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = np.moveaxis(img, -1, 0)
    img = img[None, :]
    return img

def read_pixel_data(part="train"):
    with open(f"{DATA_PATH}/{part}.txt") as f:
        lines = f.readlines()
        X, y = [], []
        for l in lines:
            img_path, label = l.split()
            img_path = f"{DATA_PATH}/{img_path}"
            X.append(img_path)
            y.append(label)
        pool = Pool(CPU_USED)
        pool_output = list(tqdm(pool.imap(img2np, X), total=len(X)))
        X = np.concatenate(pool_output, axis=0)
        y = np.array(y, dtype=int)
    return (X, y)

def one_hot_encoding(y):
    y_one = np.zeros((y.size, y.max() + 1))
    y_one[np.arange(y.size), y] = 1
    return y_one

# above by myself

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]
