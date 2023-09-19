import numpy as np
from optimizer import Optimizer
from utils import show_lineChart

class LinearRegressionMatrix:
    def __init__(self, 
                 x: np.matrix, 
                 y: np.matrix):
        bias = np.ones(x.shape[0])
        self.x = np.insert(x, x.shape[1], values=bias, axis=1)
        self.y = y
        self.w = np.linalg.pinv(self.x) @ self.y

    def predict(self, x):
        bias = np.ones(x.shape[0])
        x = np.insert(x, x.shape[1], values=bias, axis=1)
        # 此处的x是多个样本的矩阵，每一行是一个样本
        return x @ self.w
    
class LinearRegression:
    def __init__(self, 
                 optimizer: Optimizer,
                 x: np.matrix, 
                 y: np.matrix):
        bias = np.ones(x.shape[0])
        self.x = np.insert(x, x.shape[1], values=bias, axis=1)
        self.y = y
        self.w = np.matrix(np.zeros(self.x.shape[1])).T
        self.optimizer = optimizer

    def predict(self, x):
        bias = np.ones(x.shape[0])
        x = np.insert(x, x.shape[1], values=bias, axis=1)
        return x @ self.w
    
    def train(self,
              epochs: int,
              show_loss: bool = False):
        n = self.x.shape[0]
        loss_list = []
        for epoch in range(epochs):
            grad = (self.x.T @ self.x @ self.w - self.x.T @ self.y) * 2 / n
            if show_loss:
                loss = np.linalg.norm(self.x @ self.w - self.y) ** 2 / n
                loss_list.append(loss)
            self.w = self.optimizer.update(self.w, grad)
        if show_loss:
            show_lineChart(loss_list, xlabel='epoch', ylabel='loss')

# 因为随机梯度下降改变了训练过程，所以新写一个类
class LinearRegressionSDG(LinearRegression):
    def __init__(self, 
                 optimizer: Optimizer,
                 x: np.matrix, 
                 y: np.matrix,
                 batch_size: int):
        super().__init__(optimizer, x, y)
        self.batch_size = batch_size
    
    def train(self,
              epochs: int,
              show_loss: bool = False):
        n = self.x.shape[0]
        loss_list = []
        
        for epoch in range(epochs):
            # 随机打乱数据
            data = np.concatenate((self.x, self.y), axis=1)
            np.random.shuffle(data)
            for i in range(0, n, self.batch_size):
                batch = data[i:min(i + self.batch_size, n), :]
                batch_x = batch[:, :-1]
                batch_y = batch[:, -1]
                grad = (batch_x.T @ batch_x @ self.w - batch_x.T @ batch_y) * 2 / n
                # 每个batch更新一次
                self.w = self.optimizer.update(self.w, grad)
            if show_loss:
                loss = np.linalg.norm(self.x @ self.w - self.y) ** 2 / n
                loss_list.append(loss)
        if show_loss:
            show_lineChart(loss_list, xlabel='epoch', ylabel='loss')