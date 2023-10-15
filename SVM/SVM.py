import numpy as np
from cvxopt import matrix
import cvxopt

class SVM:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        self.dim = x.shape[1]
        self.n = x.shape[0]
        self.x = self.augment(x)
        self.y = y

    def quadratic_programming(self):
        Q = np.eye(1 + self.dim)
        Q[0, 0] = 0

        p = np.zeros((1 + self.dim, 1))

        # 因为cvxopt的不等式约束是 a^TU <= c ，所以加负号把不等式反过来
        a = -np.multiply(self.y, self.x)
        c = -np.ones((self.n, 1))
        c = -5 * np.ones((self.n, 1))

        Q = matrix(Q)
        p = matrix(p)
        a = matrix(a)
        c = matrix(c)

        self.w = cvxopt.solvers.qp(Q, p, a, c)['x']
        self.w = np.array(self.w)
        return self.w

    def gradient_descent(self, lr, epochs, batch_size=1):
        self.w = np.zeros((1 + self.dim, 1))
        data = np.concatenate((self.x, self.y), axis=1)
        np.random.shuffle(data)
        for epoch in range(epochs):
            for i in range(0, self.n, batch_size):
                batch = data[i:min(i + batch_size, self.n), :]
                batch_x = batch[:, :-1]
                batch_y = batch[:, -1]
                grad = self.hinge_loss_grad(self.w, batch_x, batch_y)
                self.w = self.w - lr * grad
            if np.all(1 - np.multiply(self.y, (self.x @ self.w)) <= 0):
                print('break at epoch {}'.format(epoch))
                break
        return self.w

    def hinge_loss(self, w, x, y):
        return np.maximum(0, 1 - np.multiply(y, (x @ w)))
    
    def hinge_loss_grad(self, w, x, y):
        b = x.shape[0]
        condition = np.matrix(np.zeros(y.shape))
        condition[1 - np.multiply(y, (x @ w)) > 0] = 1
        grads = np.multiply(condition, np.multiply(-y, x))
        return np.sum(grads.T, axis=1) / b
    
    @staticmethod
    def augment(x: np.ndarray) -> np.ndarray:
        x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
        return x
    
class Dual_SVM:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray):
        self.dim = x.shape[1]
        self.n = x.shape[0]
        self.x = x
        self.y = y

    def quadratic_programming(self):
        # 矩阵必须是np.ndarray类，若为np.matrix类，*不是哈达马积而是矩阵乘法
        Q = (self.x @ self.x.T) * (self.y @ self.y.T)
        p = -np.ones((self.n, 1))
        A = -np.eye(self.n)
        c = np.zeros((self.n, 1))
        # https://cvxopt.org/userguide/coneprog.html#quadratic-programming
        # cvxopt中，rx直接相乘，不转置。因此先把它转置了
        r = self.y.T
        v = np.zeros((1, 1))

        Q = matrix(Q)
        p = matrix(p)
        A = matrix(A)
        c = matrix(c)
        r = matrix(r)
        v = matrix(v)

        alpha = cvxopt.solvers.qp(Q, p, A, c, r, v)['x']
        alpha = np.array(alpha)

        self.w = (alpha * self.y).T @ self.x
        self.w = self.w[0, :]

        # 找到第一个不为0的alpha对应的样本序号
        idx = np.where(alpha>1e-9)[0][0]
        b = self.y[idx] - self.w @ self.x[idx].T

        self.w = np.concatenate((b, self.w))
