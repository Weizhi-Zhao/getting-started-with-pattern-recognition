# from SVM import SVM, Dual_SVM
# import numpy as np
# from utils import show_fig, gen_two_random_normal

# x = np.array([[1, 2], [3, 4]])
# y = np.array([[1], [-1]])
# m1 = [-5, 0]
# s1 = [[1, 0], [0, 1]]
# size1 = 20
# m2 = [0, 5]
# s2 = [[1, 0], [0, 1]]
# size2 = 20
# train_x, train_y, test_x, test_y = gen_two_random_normal(m1, s1, size1, m2, s2, size2)

# svm = Dual_SVM(train_x, train_y)
# svm.quadratic_programming()

# print(np.concatenate((np.ones((x.shape[0], 1)), x), axis=1))

# show_fig(train_x, train_y, w=svm.w)

import copy

def lala(a):
    b = copy.copy(a)
    b.a = 100
    b.c = numm(300)
    b()

    c = copy.deepcopy(a)
    c.a = 100
    c.c = numm(300)
    c()

    # a.a = 100
    # a.c = numm(300)
    a()

class numm:
    def __init__(self, x):
        self.n = x
class mc:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = numm(3)

    def __call__(self):
        print(self.a)
        print(self.b)
        print(self.c.n)

x = mc()
lala(x)
x()
