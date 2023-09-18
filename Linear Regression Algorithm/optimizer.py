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

# Adagrad
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
        self.sum_square_grad += np.square(grad)
        sigma = np.sqrt( 1/(self.t+1) *  self.sum_square_grad)
        lr_t = self.lr / np.sqrt(self.t + 1)
        self.t += 1
        return w - np.multiply(lr_t / (sigma+self.epsilon), grad)
    
# RMSProp
class RMSProp(Optimizer):
    def __init__(self, lr, alpha, epsilon=1e-6):
        super().__init__(lr)
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma = None

    def update(self,
               w: np.matrix,
               grad: np.matrix) -> np.matrix:
        if self.sigma is None:
            self.sigma = np.sqrt( np.square(grad) )
        else:
            self.sigma = np.sqrt(
                    self.alpha * self.sigma + (1 - self.alpha) * np.square(grad)
                )
        return w - np.multiply(self.lr / (self.sigma+self.epsilon), grad)
    
# Momentum
class Momentum(Optimizer):
    def __init__(self, lr, lamb):
        super().__init__(lr)
        self.lamb = lamb
        self.m = None

    def update(self,
               w: np.matrix,
               grad: np.matrix) -> np.matrix:
        if self.m is None:
            self.m = np.zeros(grad.shape)
        self.m = self.lamb * self.m - self.lr * grad
        return w + self.m

# Adam
class Adam(Optimizer):
    def __init__(self, lr, beta1, beta2, epsilon=1e-6):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None
        self.epsilon = epsilon

    def update(self,
               w: np.matrix,
               grad: np.matrix) -> np.matrix:
        if self.m is None:
            self.m = np.zeros(grad.shape)
        if self.v is None:
            self.v = np.zeros(grad.shape)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad)
        m_hat = self.m / (1 - np.power(self.beta1, self.t))
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
