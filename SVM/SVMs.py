import numpy as np
from cvxopt import matrix
import cvxopt
from typing import Callable

cvxopt.solvers.options['show_progress'] = False

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
        a = -self.y * self.x
        c = -np.ones((self.n, 1))

        Q = matrix(Q)
        p = matrix(p)
        a = matrix(a)
        c = matrix(c)

        self.w = cvxopt.solvers.qp(Q, p, a, c)['x']
        self.w = np.array(self.w).squeeze()
        return self.w

    def gradient_descent(self, lr, epochs, batch_size=1):
        self.w = np.zeros(1 + self.dim)
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
        condition = np.zeros(y.shape)
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

        idx = np.where(alpha>1e-6)[0]
        self.sv_alpha = alpha[idx]
        self.sv_x = self.x[idx]
        self.sv_y = self.y[idx]

        self.w = (alpha * self.y).T @ self.x
        self.w = self.w[0, :]

        # 找到第一个不为0的alpha对应的样本序号
        # idx = np.where(alpha>1e-9)[0][0]
        b = self.y[idx[0]] - self.w @ self.x[idx[0]].T

        self.w = np.concatenate((b, self.w))

class Kernel_SVM:
    def __init__(self,
                 x: np.ndarray,
                 y: np.ndarray,
                 kernel_func: Callable):
        self.dim = x.shape[1]
        self.n = x.shape[0]
        self.x = x
        self.y = y
        self.kernel_func = kernel_func

    def quadratic_programming(self):
        # 矩阵必须是np.ndarray类，若为np.matrix类，*不是哈达马积而是矩阵乘法
        Q = self.y @ self.y.T * self.kernel_func(self.x, self.x)
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

        # 找到所有支撑向量
        idx = np.where(alpha>1e-6)[0]

        self.sv_alpha = alpha[idx]
        self.sv_x = self.x[idx]
        self.sv_y = self.y[idx]

        # 从alpha>0的x和y中任选一个，但是它要是二维向量
        y_querry = self.sv_y[None, 0]
        x_querry = self.sv_x[None, 0]
        # self.sv_y * self.sv_alpha 是二维列向量
        b = y_querry - (self.sv_y * self.sv_alpha * self.kernel_func(self.sv_x, x_querry)).sum()
        # b变成float（去掉多余向量）
        self.b = np.squeeze(b)

    def __call__(self, x, b=None):
        wx = np.multiply(self.sv_alpha * self.sv_y, self.kernel_func(self.sv_x, x)).sum(axis=0)
        # 生成的数据中y是列向量，因此转置一下（在每个数字上插一个维度）
        if b is not None:
            return np.sign(wx + b)[:, None]
        return np.sign(wx + self.b)[:, None]

if __name__ == '__main__':
    from kernel import gauss_kernel, polynomial_kernel
    from functools import partial
    from utils import gen_two_random_normal, show_fig, plot_decision_boundary

    # 生成数据集
    m1 = [-5, 0]
    s1 = [[1, 0], [0, 1]]
    size1 = 200
    m2 = [0, 5]
    s2 = [[1, 0], [0, 1]]
    size2 = 200
    train_x, train_y, test_x, test_y = gen_two_random_normal(m1, s1, size1, m2, s2, size2)

    # SVM
    svm = SVM(train_x, train_y)
    svm.quadratic_programming()
    show_fig(test_x, test_y, w=svm.w)

    # dual SVM
    svm = Dual_SVM(train_x, train_y)
    svm.quadratic_programming()
    show_fig(test_x, test_y, w=svm.w)
    
    # kernel SVM
    kernel = partial(gauss_kernel, gamma=0.1)
    # kernel = partial(polynomial_kernel, zeta=1, gamma=0.5, order=4)
    svm = Kernel_SVM(train_x, train_y, kernel)
    svm.quadratic_programming()
    # plot_decision_boundary(train_x, svm, h=0.1)
    res = svm(test_x)
    show_fig(test_x, test_y, pred_func=svm)
