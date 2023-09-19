import numpy as np
import matplotlib.pyplot as plt
from optimizer import GradientDescent, Adagrad, RMSProp, Momentum, Adam

def train(optim, x, epochs, noise=False):
    x_list = [x]
    y_list = [x * np.cos(0.25 * np.pi * x)]
    for epoch in range(epochs):
        grad = np.cos(0.25 * np.pi * x) - np.sin(0.25 * np.pi * x) * 0.25 * np.pi * x
        if noise:
            grad += np.random.normal(0, 0.1)
        x = optim.update(x, grad)

        x_list.append(x)
        y_list.append(x * np.cos(0.25 * np.pi * x))
    return x_list, y_list

LR = 0.4
X = -4

optimizers = [GradientDescent(LR), Adagrad(LR, 1e-6), RMSProp(LR, 0.9), Momentum(LR, 0.9),
              Adam(LR, 0.9, 0.999)]

x, y = train(GradientDescent(LR), X, 200, noise=True)
plt.plot(x)
plt.plot(y)
plt.show()
print(x[-1], y[-1])

for optim in optimizers:
    x, y = train(optim, X, 20)
    plt.plot(x)
    plt.plot(y)
    plt.show()
    print(x[-1], y[-1])