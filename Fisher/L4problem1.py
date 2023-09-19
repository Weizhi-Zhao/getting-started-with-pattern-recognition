from Fisher import Fisher
import numpy as np
from utils import show_fig

c1 = np.matrix(
        [
            [5, 37],
            [7, 30],
            [10, 35],
            [11.5, 40],
            [14, 38],
            [12, 31]
        ]
    )

c2 = np.matrix(
        [
            [35, 21.5],
            [39, 21.7],
            [34, 16],
            [37, 17]
        ]
    )

fisher = Fisher(c1, c2)

y = np.concatenate((np.ones(c1.shape[0]), -np.ones(c2.shape[0])))
x = np.concatenate((c1, c2))
y_predict = fisher.predict(x)

show_fig(x, y, y_predict, fisher.w, fisher.threshold)

print(fisher.w, fisher.u1, fisher.u2, fisher.sigma1, fisher.sigma2, 
      np.linalg.inv(fisher.sw), fisher.threshold)
