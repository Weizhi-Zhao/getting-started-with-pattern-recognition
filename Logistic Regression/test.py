from logistic_regression import LogisticRegression
import numpy as np
from utils import gen_two_random_normal, show_fig

x = np.array([
              [1, 2, 3],
              [2, 3, 7],
              [3, 9, 2]
             ])
y = x[:, 1:3]

print(y)
