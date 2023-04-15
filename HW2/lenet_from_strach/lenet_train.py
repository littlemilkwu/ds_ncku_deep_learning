import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser

from config import *
from model_layer import *
from toolbox import *

parser = ArgumentParser()
parser.add_argument("-v", "--version", default='origin', type=str)
parser.add_argument("-e", "--epoch", default=EPOCHS, type=int)
parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
VERSION = args.version

def preprocessing():
    train_X, train_y = read_pixel_data(part='train')
    val_X, val_y = read_pixel_data(part='val')
    train_X, val_X = train_X / float(255), val_X / float(255)
    train_X -= np.mean(train_X)
    val_X -= np.mean(val_X)
    train_y = one_hot_encoding(train_y)
    val_y = one_hot_encoding(val_y)
    return (train_X, train_y), (val_X, val_y)

def train_loop(model, train_X, train_y, val_X, val_y, loss_fn, optim):
    ls_loss = []
    best_val_loss = 1e10
    for e in range(EPOCHS):
        batch_cnt = train_X.shape[0] // BATCH_SIZE
        bar = tqdm(range(batch_cnt), desc=f'[Epoch {e}] ')
        train_X, train_y = union_shuffle(train_X, train_y)

        train_loss = 0
        train_acc = 0
        for batch_i in bar:
            X = train_X[batch_i*BATCH_SIZE: (batch_i+1)*BATCH_SIZE]
            y= train_y[batch_i*BATCH_SIZE: (batch_i+1)*BATCH_SIZE]
            y_pred = model.forward(X)
            loss, dout = loss_fn.get(y_pred, y)
            train_loss += loss
            train_acc += (y_pred.argmax(axis=1) == y.argmax(axis=1)).sum() / BATCH_SIZE
            model.backward(dout)
            optim.step()
            if batch_i == batch_cnt - 1:
                val_y_pred = model.forward(val_X)
                val_loss, dout = loss_fn.get(val_y_pred, val_y)
                val_acc = (val_y_pred.argmax(axis=1) == val_y.argmax(axis=1)).sum() / val_y.shape[0]
                bar.set_postfix(train_loss=train_loss/batch_cnt, train_acc=train_acc/batch_cnt, val_loss=val_loss, val_acc=val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_model(model)

        train_loss /= batch_cnt
        train_acc /= batch_cnt
        ls_loss.append([train_loss, train_acc, val_loss, val_acc])
    return ls_loss

def main():
    s_time = time.time()
    (train_X, train_y), (val_X, val_y) = preprocessing()
    if VERSION == 'improved':
        model = LeNet5Imp()
    elif VERSION == 'origin':
        model = LeNet5()
    optim = SGDMomentum(model.get_params(), lr=1e-3, momentum=0.80, reg=0.00003)
    loss_fn = CrossEntropyLoss()
    ls_loss = train_loop(model, train_X, train_y, val_X, val_y, loss_fn, optim)

    dict_hyper = {"v": VERSION, "e": EPOCHS, "b": BATCH_SIZE}
    save_loss(ls_loss, dict_hyper)
    save_model(model, dict_hyper)

    print(f'total spent: {int(time.time() - s_time)} secs.')
    # time_start = time.time()

    # train_X = train_X[:BATCH_SIZE]
    # train_y = train_y[:BATCH_SIZE]
    
    # y_pred = model.forward(train_X)
    # print(f'forward used: {time.time() - time_start}')
    # loss, dout = loss_fn.get(y_pred, train_y)
    # model.backward(dout)
    # optim.step()

    # time_end = time.time()
    # print(f'one batch spend {round(time_end - time_start, 2)}')
def test():
    conv1 = Conv(1, 6, 5)
    np_ar1 = np.arange(98).reshape(2, 1, 7, 7)
    print('ar1: ', np_ar1)

    np_ar2 = conv1._forward(np_ar1)
    # print('ar2: ', np_ar2)

if __name__ == '__main__':
    # test()
    main()