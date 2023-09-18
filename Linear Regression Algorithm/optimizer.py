import numpy as np

class Optimizer:
    def __init__(self,
                 lr: float):
        self.lr = lr

    def update(self,
               w: np.matrix,
               grad: np.matrix) -> np.matrix:
        pass

# 梯度下降法
class GradientDescent(Optimizer):
    def __init__(self,
                 lr: float):
        super().__init__(lr)

    def update(self,
               w: np.matrix,
               grad: np.matrix) -> np.matrix:
        return w - self.lr * grad

class Adagrad(Optimizer):
    def __init__(self,
                 lr: float,
                 epsilon: float = 1e-6):
        super().__init__(lr)
        self.epsilon = epsilon
        self.sum_square_grad = None
        self.t = 0

    def update(self,
               w: np.matrix,
               grad: np.matrix) -> np.matrix:
        if self.sum_square_grad is None:
            self.sum_square_grad = np.zeros(grad.shape)
        # 对应元素相乘
        self.sum_square_grad += np.multiply(grad, grad)
        sigma = np.sqrt( 1/(self.t+1) *  self.sum_square_grad)
        lr_t = self.lr / np.sqrt(self.t + 1)
        self.t += 1
        return w - np.multiply(lr_t / sigma, grad)