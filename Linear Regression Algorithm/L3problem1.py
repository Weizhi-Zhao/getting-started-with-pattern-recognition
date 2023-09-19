import numpy as np
from linear_regression import LinearRegressionMatrix, LinearRegression, LinearRegressionSDG
from optimizer import GradientDescent
from utils import show_fig

x = np.matrix(
        [
            [0.2, 0.7],
            [0.3, 0.3],
            [0.4, 0.5],
            [0.6, 0.5],
            [0.1, 0.4],
            [0.4, 0.6],
            [0.6, 0.2],
            [0.7, 0.4],
            [0.8, 0.6],
            [0.7, 0.5]
        ]
    )
y = np.matrix([1, 1, 1, 1, 1, -1, -1, -1, -1, -1]).T

LRM = LinearRegressionMatrix(x, y)

optim = GradientDescent(lr=0.1)
LRG = LinearRegressionSDG(optim, x, y, batch_size=1)
LRG.train(10000)

print(LRG.w)
print(LRM.w)

print(np.sign(LRG.predict(x)))
print(np.sign(LRM.predict(x)))

show_fig(x, y, LRG.w)