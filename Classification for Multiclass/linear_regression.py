import numpy as np

class LinearRegressionMatrix:
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray):
        # bias = np.ones(x.shape[0])
        # self.x = np.insert(x, x.shape[1], values=bias, axis=1)
        self.x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        self.y = y
        self.w = np.linalg.pinv(self.x) @ self.y
        self.w = self.w.squeeze()

    def predict(self, x):
        # bias = np.ones(x.shape[0])
        # x = np.insert(x, x.shape[1], values=bias, axis=1)
        # import pdb; pdb.set_trace()
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        # 此处的x是多个样本的矩阵，每一行是一个样本
        return x @ self.w
