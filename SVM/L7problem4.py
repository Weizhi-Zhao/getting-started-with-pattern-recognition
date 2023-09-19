from SVM import SVM
import numpy as np
from utils import show_fig, gen_two_random_normal

x = np.matrix([
        [1, 1],
        [2, 2],
        [2, 0],
        [0, 0],
        [1, 0],
        [0, 1]
    ])
y = np.matrix([1, 1, 1, -1, -1, -1]).T

svm = SVM(x, y)
# svm.quadratic_programming()
svm.gradient_descent(lr=0.01, epochs=1000, batch_size=1)

show_fig(x, y, w=svm.w)
print(svm.w)
