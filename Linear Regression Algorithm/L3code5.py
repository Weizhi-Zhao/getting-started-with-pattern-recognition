import numpy as np
import matplotlib.pyplot as plt
from optimizer import GradientDescent, Adagrad

def train(optim, x, epochs):
    x_list = [x]
    y_list = [x * np.cos(0.25 * np.pi * x)]
    for epoch in range(epochs):
        grad = np.cos(0.25 * np.pi * x) - np.sin(0.25 * np.pi * x) * 0.25 * np.pi * x
        x = optim.update(x, grad)

        x_list.append(x)
        y_list.append(x * np.cos(0.25 * np.pi * x))
    return x_list, y_list

LR = 0.4
X = -4

x, y = train(GradientDescent(LR), X, 10)
plt.plot(x)
plt.plot(y)
plt.show()

x, y = train(Adagrad(LR, 1e-6), X, 10)
plt.plot(x)
plt.plot(y)
plt.show()