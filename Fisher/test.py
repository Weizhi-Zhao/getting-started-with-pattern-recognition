from Fisher import Fisher
import numpy as np
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

c1 = np.matrix(
        [
            [0.2, 0.7],
            [0.3, 0.3],
            [0.4, 0.5],
            [0.6, 0.5],
            [0.1, 0.4]
        ]
    )

c2 = np.matrix(
        [
            [0.4, 0.6],
            [0.6, 0.2],
            [0.7, 0.4],
            [0.8, 0.6],
            [0.7, 0.5]
        ]
    )

fisher = Fisher(c1, c2)

y = np.concatenate((np.ones(c1.shape[0]), -np.ones(c2.shape[0])))
x = np.concatenate((c1, c2))
y_predict = fisher.predict(x)

show_fig(x, y, y_predict, fisher.w, fisher.threshold)

print(fisher.w, fisher.u1, fisher.u2, fisher.sigma1, fisher.sigma2, 
      np.linalg.inv(fisher.sw), fisher.threshold)
