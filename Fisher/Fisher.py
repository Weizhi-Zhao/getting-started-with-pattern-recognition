import numpy as np

class Fisher:
    def __init__(self, 
                 c1: np.matrix,
                 c2: np.matrix):
        self.c1 = c1
        self.c2 = c2

        self.u1 = np.mean(c1, axis=0).T
        self.u2 = np.mean(c2, axis=0).T

        self.sigma1 = np.cov(c1, rowvar=False)
        self.sigma2 = np.cov(c2, rowvar=False)

        self.sw = self.sigma1 + self.sigma2
        self.w = np.linalg.inv(self.sw) @ (self.u1 - self.u2)

        self.threshold = self.w.T @ (self.u1 + self.u2) / 2

    def predict(self, x):
        return np.sign(x @ self.w - self.threshold)
