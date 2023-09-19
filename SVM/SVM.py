import numpy as np
from cvxopt import matrix
import cvxopt

class SVM:
    def __init__(self,
                 x: np.matrix,
                 y: np.matrix,):
        self.dim = x.shape[1]
        self.n = x.shape[0]
        bias = np.ones((x.shape[0], 1))
        self.x = np.concatenate((bias, x), axis=1)
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