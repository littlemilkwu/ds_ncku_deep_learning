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
parser.add_argument("-e", "--epochs", default=EPOCHS, type=int)
parser.add_argument("-b", "--batch_size", default=BATCH_SIZE, type=int)
parser.add_argument("-lr", "--learning_rate", default=LEARNING_RATE, type=float)
parser.add_argument("-m", "--mode", default='official', type=str)
args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
VERSION = args.version
MODE = args.mode
dict_hyper = {"v": VERSION, "e": EPOCHS, "b": BATCH_SIZE, 'lr': LEARNING_RATE}

def preprocessing():
    train_X, train_y = read_pixel_data(part='train')
    val_X, val_y = read_pixel_data(part='val')
    train_X, val_X = train_X / float(255), val_X / float(255)
    train_X -= np.mean(train_X)
    val_X -= np.mean(val_X)
    train_y = one_hot_encoding(train_y, 50)
    val_y = one_hot_encoding(val_y, 50)
    return (train_X, train_y), (val_X, val_y)

def test_mode(train_X, train_y):
    train_X, train_y = union_shuffle(train_X, train_y)
    train_X, train_y = train_X[:10000], train_y[:10000]
    return train_X, train_y

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
                    save_model(model, dict_hyper)

        train_loss /= batch_cnt
        train_acc /= batch_cnt
        ls_loss.append([train_loss, train_acc, val_loss, val_acc])
    return ls_loss

def main():
    global VERSION, MODE, EPOCHS, BATCH_SIZE, LEARNING_RATE
    s_time = time.time()

    (train_X, train_y), (val_X, val_y) = preprocessing()
    if VERSION == 'improved':
        model = LeNet5Imp()
    elif VERSION == 'origin':
        model = LeNet5()

    if MODE == "test":
        train_X, train_y = test_mode(train_X, train_y)
        dict_hyper['v'] = 'test_' + dict_hyper['v']

    print(f"#"*30)
    print(f"{'VERSION':<10}: {VERSION}")
    print(f"{'MODE':<10}: {MODE}")
    print(f"{'EPOCHS':<10}: {EPOCHS}")
    print(f"{'BATCH SIZE':<10}: {BATCH_SIZE}")
    print(f"{'LR':<10}: {LEARNING_RATE}")
    print(f"{'TRAIN NUM':<10}: {train_X.shape[0]}")
    print(f"#"*30)

    optim = SGDMomentum(model.get_params(), lr=LEARNING_RATE, momentum=0.80, reg=0.00003)
    loss_fn = CrossEntropyLoss()
    ls_loss = train_loop(model, train_X, train_y, val_X, val_y, loss_fn, optim)

    save_loss(ls_loss, dict_hyper)
    print(f'total spent: {int(time.time() - s_time)} secs.')

if __name__ == '__main__':
    main()