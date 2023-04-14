import os
import cv2
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from config import *
from toolbox import *

class Tensor:
    def __init__(self, nparr:np.ndarray):
        self.value = nparr
        self.grad = np.zeros(shape=nparr.shape)
        self.back = None

class Multiply_:
    def __init__(self):
        self.gra_local = 0

    def __call__(self, x1:Tensor, x2:Tensor):
        self.x1 = x1
        self.x2 = x2
        self.out = Tensor(np.dot(x1.value, x2.value))
        self.out.back = self
        return self.out
    
    def backward(self, gra_up):
        self.gra_up = gra_up
        
        self.x1.grad = np.dot(gra_up, self.x2.value.T)
        self.x2.grad = np.dot(self.x1.value.T, gra_up)

        if self.x1.back != None:
            self.x1.back.backward(self.x1.grad)
        if self.x2.back != None:
            self.x2.back.backward(self.x2.grad)

class Sum_:
    def __init__(self):
        self.gra_local = 0
    
    def __call__(self, x1:Tensor, x2:Tensor):
        self.x1 = x1
        self.x2 = x2
        self.out = Tensor(x1.value + x2.value)
        self.out.back = self
        return self.out
    
    def backward(self, gra_up):
        self.x1.grad = gra_up
        self.x2.grad = gra_up.mean(axis=0)
        # print('sum grad: ', self.x2.grad)
        if self.x1.back != None:
            self.x1.back.backward(self.x1.grad)
        if self.x2.back != None:
            self.x2.back.backward(self.x2.grad)

class Relu_:
    def __init__(self):
        self.gra_local = 0
    
    def __call__(self, x:Tensor):
        self.x = x
        self.out = Tensor(np.maximum(self.x.value, 0))
        self.out.back = self
        return self.out
    
    def backward(self, gra_up):
        self.x.grad = np.float64(gra_up > 0) * gra_up
        
        if self.x.back != None:
            self.x.back.backward(self.x.grad)
    
class CrossEntropy_:
    def __init__(self):
        self.gra_local = 1

    def __call__(self, y_pred:Tensor, y):
        self.y_pred = y_pred
        self.y = y
        # softmax
        self.s = (np.exp(y_pred.value).T / np.exp(y_pred.value).sum(axis=1)).T

        # cross-entropy
        self.out = -np.sum(np.log(self.s)*y)
        return self.out
    
    def backward(self):
        self.gra_local = self.s - self.y
        self.y_pred.grad = self.gra_local
        if self.y_pred.back != None:
            self.y_pred.back.backward(self.y_pred.grad)
        return self.gra_local

class ComGraph:
    def __init__(self):
        limit = np.sqrt(1 / (INPUT_DIM * 3))
        self.W1 = Tensor(np.random.uniform(low=-limit, high=limit, size=(INPUT_DIM, HIDDEN_DIM)))
        self.b1 = Tensor(np.random.uniform(low=-limit, high=limit, size=(HIDDEN_DIM, )))
        self.mult1 = Multiply_()
        self.sum1 = Sum_()
        self.relu1 = Relu_()

        self.W2 = Tensor(np.random.uniform(low=-limit, high=limit, size=(HIDDEN_DIM, OUTPUT_DIM)))
        self.b2 = Tensor(np.random.uniform(low=-limit, high=limit, size=(OUTPUT_DIM, )))
        self.mult2 = Multiply_()
        self.sum2 = Sum_()

    def __call__(self, X):
        self.h1 = self.mult1(X, self.W1) # (128, 256)
        self.h2 = self.sum1(self.h1, self.b1)
        self.h3 = self.relu1(self.h2)
        self.h4 = self.mult2(self.h3, self.W2) # (128, 50)
        self.out = self.sum2(self.h4, self.b2)
        # print('forward out: ', self.out.value)
        return self.out
    
    def step(self, lr):
        self.W1.value = self.W1.value - self.W1.grad * lr
        self.W2.value = self.W2.value - self.W2.grad * lr
        self.b1.value = self.b1.value - self.b1.grad * lr
        self.b2.value = self.b2.value - self.b2.grad * lr
        pass


def train_loop(model, train_X, train_y, val_X, val_y, loss_fn):
    batch_cnt = train_X.shape[0] // BATCH_SIZE
    ls_loss = []
    for e in range(1, EPOCHS + 1):
        bar = tqdm(range(batch_cnt), desc=f'[Epochs {e:2d}]')
        epoch_loss = 0
        epoch_acc = 0
        for batch_i in bar:
            X = train_X[batch_i * BATCH_SIZE: (batch_i + 1) * BATCH_SIZE]
            y = train_y[batch_i * BATCH_SIZE: (batch_i + 1) * BATCH_SIZE]
            X = Tensor(X)
            y_pred = model(X)
            loss = loss_fn(y_pred, y) / BATCH_SIZE
            acc = (y_pred.value.argmax(axis=1) == y.argmax(axis=1)).sum() / BATCH_SIZE
            bar.set_postfix(loss=loss, acc=acc)
            epoch_loss += loss
            epoch_acc += acc

            loss_fn.backward()
            model.step(LEARNING_RATE)
            
            
            print(model.h1.value[:10, :3])
            
            if batch_i == batch_cnt - 1:
                # validate
                val_y_pred = model(val_X)
                val_loss = loss_fn(val_y_pred, val_y) / len(val_y)
                val_acc = (val_y_pred.value.argmax(axis=1) == val_y.argmax(axis=1)).sum() / len(val_y)
                bar.set_postfix(loss=loss, acc=acc, val_loss=val_loss, val_acc=val_acc)
        epoch_loss /= batch_cnt
        epoch_acc /= batch_cnt
        ls_loss.append([epoch_loss, epoch_acc, val_loss, val_acc])
    return ls_loss

def main():
    # train_X, train_y = read_pixel_data(part='val')
    train_X, train_y = read_img_feature_data(part='val')
    val_X, val_y = read_img_feature_data(part='val')

    # preprocessing
    train_X, train_y = union_shuffle(train_X, train_y)
    train_y = one_hot_encoding(train_y)
    val_y = one_hot_encoding(val_y)
    val_X = Tensor(val_X)

    # build model
    model = ComGraph()
    loss_fn = CrossEntropy_()
    ls_loss = train_loop(model, train_X, train_y, val_X, val_y, loss_fn)

    pd.DataFrame(ls_loss, columns=['train_loss', 'train_acc', 'val_loss', 'val_acc'])\
        .to_csv("output/ComGraph_result.csv", index=False)
    return 


if __name__ == "__main__":
    main()