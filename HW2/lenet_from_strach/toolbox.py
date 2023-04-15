import gzip
import pickle
import random
import cv2
import numpy as np
import pandas as pd
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

def one_hot_encoding(y, class_num):
    y_one = np.zeros((y.size, class_num))
    y_one[np.arange(y.size), y] = 1
    return y_one

def save_loss(ls_loss, dict_hyper:dict):
    pd.DataFrame(ls_loss, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])\
        .to_csv(f'output/lenet_{dict_hyper["v"]}_e{dict_hyper["e"]}_b{dict_hyper["b"]}.csv', index=False)

def save_model(model, dict_hyper:dict):
    weights = model.get_params()
    with open(f"model_weights/lenet_{dict_hyper['v']}_e{dict_hyper['e']}_b{dict_hyper['b']}_weights.pkl","wb") as f:
        pickle.dump(weights, f)

def draw_loss_n_save(ls_loss, dict_hyper):
    train_loss, train_acc, val_loss, val_acc = zip(*ls_loss)
    epochs = list(range(1, len(train_loss) + 1) )
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    fig.suptitle(f"{dict_hyper['v']} lenet e{dict_hyper['e']} b{dict_hyper['b']}")
    ax[0].plot(epochs, train_loss, label='train_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].set_xticks(epochs)
    ax[0].legend()

    ax[1].plot(epochs, train_acc, label='train_acc')
    ax[1].plot(epochs, val_acc, label='val_acc')
    ax[1].set_xticks(epochs)
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    plt.tight_layout()
    fig.savefig('test.png')

# above by myself

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]
