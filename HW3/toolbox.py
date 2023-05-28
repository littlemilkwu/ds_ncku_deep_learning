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
        .to_csv(f'output/lenet_{dict_hyper["v"]}_e{dict_hyper["e"]}_b{dict_hyper["b"]}_lr{dict_hyper["lr"]}.csv', index=False)

def save_model(model, dict_hyper:dict):
    weights = model.get_params()
    with open(f"model_weights/lenet_{dict_hyper['v']}_e{dict_hyper['e']}_b{dict_hyper['b']}_lr{dict_hyper['lr']}_weights.pkl","wb") as f:
        pickle.dump(weights, f)

def draw_loss_n_save(ls_loss, dict_hyper):
    train_loss, train_acc, val_loss, val_acc = zip(*ls_loss)
    val_loss = np.array(val_loss)
    epochs = list(range(1, len(train_loss) + 1) )
    ls_xticks = [1] + [i for i in epochs if i % 10 == 0]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    fig.suptitle(f"{dict_hyper['v'].capitalize()} LeNet Epoch{dict_hyper['e']} BatchSize{dict_hyper['b']} LR{dict_hyper['lr']}")
    ax[0].plot(epochs, train_loss, label='train_loss')
    ax[0].plot(epochs, val_loss, label='val_loss')
    ax[0].legend()
    val_min_epoch = val_loss.argmin() + 1
    ax[0].set_xticks(ls_xticks + [val_min_epoch])
    ax[0].axvline(x=val_min_epoch, linestyle='dashed', color='black')

    ax[1].plot(epochs, train_acc, label='train_acc')
    ax[1].plot(epochs, val_acc, label='val_acc')
    ax[1].set_xticks(ls_xticks + [val_min_epoch])
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].axvline(x=val_min_epoch, linestyle='dashed', color='black')
    plt.tight_layout()
    fig.savefig(f"{OUT_PATH}/{dict_hyper['v'].capitalize()} LeNet Epoch{dict_hyper['e']} BatchSize{dict_hyper['b']} LR{dict_hyper['lr']}.png")

# above by myself

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

if __name__ == "__main__":
    ls_result = [[0.0153, 0.0213, 0.0086, 0.0267],
                [0.0148, 0.0462, 0.0082, 0.0689],
                [0.0142, 0.0677, 0.008, 0.0822],
                [0.0139, 0.0818, 0.0079, 0.0822],
                [0.0138, 0.0875, 0.0078, 0.0956],
                [0.0136, 0.096, 0.0077, 0.0756],
                [0.0134, 0.1035, 0.0076, 0.1156],
                [0.0132, 0.1133, 0.0075, 0.1133],
                [0.013, 0.1217, 0.0075, 0.1067],
                [0.0129, 0.1279, 0.0074, 0.1133],
                [0.0128, 0.1314, 0.0074, 0.1156],
                [0.0128, 0.1336, 0.0073, 0.1222],
                [0.0127, 0.1373, 0.0073, 0.1156],
                [0.0127, 0.1401, 0.0073, 0.1378],
                [0.0126, 0.1427, 0.0073, 0.1267],
                [0.0126, 0.1465, 0.0072, 0.12],
                [0.0125, 0.1456, 0.0072, 0.1267],
                [0.0125, 0.1507, 0.0072, 0.1333],
                [0.0125, 0.1493, 0.0072, 0.1244],
                [0.0124, 0.1546, 0.0072, 0.1222],
                [0.0124, 0.1549, 0.0072, 0.1244],
                [0.0123, 0.1571, 0.0071, 0.1333],
                [0.0123, 0.1596, 0.0071, 0.1467],
                [0.0122, 0.1619, 0.0072, 0.1289],
                [0.0122, 0.1649, 0.0071, 0.1467],
                [0.0121, 0.169, 0.0071, 0.1333],
                [0.0121, 0.1704, 0.0071, 0.1356],
                [0.012, 0.1745, 0.007, 0.1533],
                [0.012, 0.178, 0.0071, 0.1489],
                [0.0119, 0.1789, 0.0071, 0.1467],
                [0.0119, 0.1822, 0.007, 0.1489],
                [0.0118, 0.1841, 0.007, 0.1511],
                [0.0118, 0.1853, 0.0069, 0.1578],
                [0.0118, 0.1871, 0.007, 0.1511],
                [0.0117, 0.1888, 0.0069, 0.1578],
                [0.0117, 0.1913, 0.0069, 0.1467],
                [0.0117, 0.1918, 0.0068, 0.16],
                [0.0116, 0.1936, 0.0069, 0.1556],
                [0.0116, 0.1948, 0.0068, 0.1733],
                [0.0116, 0.1986, 0.0069, 0.1733],
                [0.0116, 0.2, 0.0068, 0.1644],
                [0.0115, 0.1994, 0.0068, 0.1822],
                [0.0115, 0.2023, 0.0068, 0.1756],
                [0.0115, 0.2042, 0.0068, 0.1711],
                [0.0114, 0.2059, 0.0068, 0.1667],
                [0.0114, 0.2074, 0.0068, 0.1689],
                [0.0114, 0.2088, 0.0067, 0.1911],
                [0.0114, 0.2097, 0.0068, 0.1733],
                [0.0114, 0.2115, 0.0068, 0.1867],
                [0.0113, 0.2124, 0.0067, 0.1711],
                [0.0113, 0.2137, 0.0067, 0.1711],
                [0.0113, 0.216, 0.0067, 0.1711],
                [0.0113, 0.2171, 0.0067, 0.1711],
                [0.0112, 0.2201, 0.0067, 0.1867],
                [0.0112, 0.2198, 0.0066, 0.1889],
                [0.0112, 0.2216, 0.0067, 0.1756],
                [0.0112, 0.2217, 0.0066, 0.1756],
                [0.0111, 0.2247, 0.0067, 0.1822],
                [0.0111, 0.2267, 0.0067, 0.1733],
                [0.0111, 0.2273, 0.0066, 0.1689],
                [0.0111, 0.2288, 0.0066, 0.1933],
                [0.011, 0.2316, 0.0066, 0.1889],
                [0.011, 0.2322, 0.0066, 0.1933],
                [0.011, 0.2328, 0.0066, 0.1889],
                [0.011, 0.2339, 0.0066, 0.1778],
                [0.0109, 0.2352, 0.0066, 0.1889],
                [0.0109, 0.2372, 0.0065, 0.1756],
                [0.0109, 0.2382, 0.0065, 0.1844],
                [0.0109, 0.2388, 0.0066, 0.1911],
                [0.0109, 0.24, 0.0065, 0.1822],
                [0.0108, 0.2415, 0.0066, 0.1867],
                [0.0108, 0.2435, 0.0065, 0.1956],
                [0.0108, 0.2445, 0.0065, 0.1933],
                [0.0108, 0.2449, 0.0066, 0.1978],
                [0.0108, 0.2456, 0.0065, 0.1844],
                [0.0107, 0.2492, 0.0065, 0.1956],
                [0.0107, 0.2497, 0.0065, 0.2178],
                [0.0107, 0.25, 0.0065, 0.1889],
                [0.0107, 0.2504, 0.0065, 0.1911],
                [0.0107, 0.2511, 0.0065, 0.2111],
                [0.0107, 0.2533, 0.0065, 0.1911],
                [0.0106, 0.2529, 0.0065, 0.2044],
                [0.0106, 0.2544, 0.0065, 0.18],
                [0.0106, 0.2572, 0.0065, 0.1844],
                [0.0106, 0.2586, 0.0064, 0.2],
                [0.0106, 0.2578, 0.0064, 0.2],
                [0.0105, 0.2602, 0.0064, 0.2],
                [0.0105, 0.2621, 0.0064, 0.1978],
                [0.0105, 0.2625, 0.0064, 0.2],
                [0.0105, 0.2641, 0.0064, 0.1956],
                [0.0105, 0.2633, 0.0064, 0.2],
                [0.0105, 0.2659, 0.0064, 0.2022],
                [0.0104, 0.2656, 0.0064, 0.2],
                [0.0104, 0.2671, 0.0064, 0.2],
                [0.0104, 0.2678, 0.0064, 0.2067],
                [0.0104, 0.2694, 0.0064, 0.1956],
                [0.0104, 0.2711, 0.0064, 0.2022],
                [0.0104, 0.2717, 0.0064, 0.2067],
                [0.0103, 0.2721, 0.0064, 0.2133],
                [0.0103, 0.2728, 0.0064, 0.2156]]
    dict_hyper = {'v': '', 'e': EPOCHS, 'b': BATCH_SIZE, 'lr': LEARNING_RATE}
    draw_loss_n_save(ls_result, dict_hyper)
