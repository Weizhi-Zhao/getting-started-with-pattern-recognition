import numpy as np
from typing import List

class LogisticRegression:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray) -> None:
        self.x = self.augment(x)
        self.y = y
        self.w = np.zeros(self.x.shape[1])
    
    def SDG(self, lr, epochs, batch_size=1) -> List:
        losses = []
        n = self.x.shape[0]
        for epoch in range(epochs):
            # 随机打乱数据
            data = np.concatenate((self.x, self.y), axis=1)
            np.random.shuffle(data)
            # 每个batch单独计算
            for i in range(0, n, batch_size):
                batch = data[i:min(i+batch_size, n), :]
                batch_x = batch[:, :-1]
                batch_y = batch[:, -1]
                # 计算梯度
                # 因为切片出来的batch_y直接是行向量，可以直接送进函数里
                grad = self.cross_entropy_grad(batch_x, batch_y, self.w)
                self.w = self.w - lr * grad
            # split出来的y是列向量，所以在函数里要转置
            losses.append(self.cross_entropy_loss(self.x, self.y, self.w))
        return losses

    @staticmethod
    def augment(x: np.ndarray) -> np.ndarray:
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return x
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    # 针对逻辑斯蒂回归的交叉熵
    @staticmethod
    def cross_entropy_loss(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        # 注意，y是列向量，要想使*表示逐一相乘，需要转置成行向量
        loss = np.sum( np.log( 1 + np.exp(-y.T * (w @ x.T)) ) ) / x.shape[0]
        return loss
    
    @staticmethod
    def cross_entropy_grad(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
        # 计算梯度
        grad = LogisticRegression.sigmoid(-y * (w @ x.T)) * (-y.T * x.T)
        # 此时grad是3*n的矩阵，第二维（行）是batch size
        grad = np.sum(grad, axis=1) / x.shape[0]
        return grad
