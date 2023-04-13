import os
import cv2
import time
import numpy as np
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
        self.gra_up = 0

    def __call__(self, x1:Tensor, x2:Tensor):
        self.x1 = x1
        self.x2 = x2
        self.out = Tensor(np.dot(x1.value, x2.value))
        self.out.back = self
        return self.out
    
    def backward(self, gra_up):
        self.gra_up = gra_up
        self.x1.grad = np.dot(self.gra_up, self.x2.value.T)
        self.x2.grad = np.dot(self.x1.value.T, self.gra_up)

        if self.x1.back != None:
            self.x1.back.backward(self.x1.grad)
        if self.x2.back != None:
            self.x2.back.backward(self.x2.grad)
        return self.gra_up

class Sum_:
    def __init__(self):
        self.gra_up = 0
    
    def __call__(self, x1:Tensor, x2:Tensor):
        self.x1 = x1
        self.x2 = x2
        self.out = Tensor(x1.value + x2.value)
        self.out.back = self
        return self.out
    
    def backward(self, gra_up):
        self.gra_down = gra_up
        self.x1.grad = self.gra_down
        self.x2.grad = self.gra_down
        # print('sum x1 shape: ', self.x1.grad.shape)
        # print('sum x1 shape: ', self.x2.grad.shape)
        # print(self.x1.grad)
        # print(self.x2.grad)
        if self.x1.back != None:
            self.x1.back.backward(self.gra_down)
        if self.x2.back != None:
            self.x2.back.backward(self.gra_down)
        return self.gra_down

class Relu_:
    def __init__(self):
        self.gra_up = 0
    
    def __call__(self, x:Tensor):
        self.x = x
        self.out = Tensor(np.maximum(x.value, 0))
        self.out.back = self
        return self.out
    
    def backward(self, gra_up):
        self.gra_up = gra_up
        self.x.grad = (self.gra_up > 0).astype(float)
        
        if self.x.back != None:
            self.x.back.backward(self.x.grad)
        return self.gra_up
    
class CrossEntropy_:
    def __init__(self):
        self.gra_up = 1

    def __call__(self, x:Tensor, y):
        self.x = x
        self.y = y
        # softmax
        print(np.exp(x.value))
        self.s = (np.exp(x.value) / np.exp(x.value).sum(axis=1))
        self.s = self.s.T

        # cross-entropy
        self.out = -np.sum(np.log(self.s)*y)
        return self.out
    
    def backward(self):
        self.gra_down = self.s - self.y
        # print('Cross: ')
        # print(self.gra_down)
        if self.x.back != None:
            self.x.back.backward(self.gra_down)
        return self.gra_down

class ComGraph:
    def __init__(self):
        self.W1 = Tensor(np.ones(shape=(1344, 256)) * 0.001)
        self.b1 = Tensor(np.ones(shape=(256, )) * 0.001)
        self.mult1 = Multiply_()
        self.sum1 = Sum_()
        self.relu1 = Relu_()

        self.W2 = Tensor(np.ones(shape=(256, 50)) * 0.001)
        self.b2 = Tensor(np.ones(shape=(50, )) * 0.001)
        self.mult2 = Multiply_()
        self.sum2 = Sum_()

        self.loss_fn = CrossEntropy_()

    def __call__(self, X):
        
        self.h1 = self.mult1(X, self.W1) # (128, 256)
        self.h2 = self.sum1(self.h1, self.b1)
        self.h3 = self.relu1(self.h2)
        self.h4 = self.mult2(self.h3, self.W2) # (128, 50)
        self.out = self.sum2(self.h4, self.b2)
        # print('forward out: ', self.out.value)
        return self.out

    def calc_loss(self, y):
        # loss
        self.loss = self.loss_fn(self.out, y)
        print('loss: ', self.loss)
        return self.loss

    def backward(self):
        up_grad = self.loss_fn.backward()

def main():
    # train_X, train_y = read_pixel_data(part='val')
    train_X, train_y = read_img_feature_data(part='val')
    train_X, train_y = train_X[:10], train_y[:10]

    # preprocessing
    train_y = one_hot_encoding(train_y)
    train_X = Tensor(train_X)

    model = ComGraph()
    y_pred = model(train_X)
    loss = model.calc_loss(train_y)
    model.backward()
    # print(model.W2.grad)
    return

if __name__ == "__main__":
    main()