import numpy as np
from utils import gen_two_random_normal, show_fig, show_lineChart
from logistic_regression import LogisticRegression

m1 = [-5, 0]
s1 = [[1, 0], [0, 1]]
size1 = 200
m2 = [0, 5]
s2 = [[1, 0], [0, 1]]
size2 = 200
train_x, train_y, test_x, test_y = gen_two_random_normal(m1, s1, size1, m2, s2, size2)

LR = LogisticRegression(train_x, train_y)

LEARNING_RATE = 0.1
BATCH_SIZE = 1
EPOCH = 30

losses = LR.SDG(LEARNING_RATE, EPOCH, BATCH_SIZE)

show_fig(train_x, train_y, LR.w)
show_lineChart(losses)
print("weight:{}".format(LR.w))
print("final loss:{}".format(losses[-1]))
print('weight norm:{}'.format(np.linalg.norm(LR.w)))

posibility = LR.sigmoid(test_y.T * (LR.w @ LR.augment(test_x).T))
posibility = np.squeeze(posibility)
print(posibility)
print(posibility.shape)

for x, y, p in zip(test_x, test_y, posibility):
    print("x:{}, y:{}, p:{}".format(x, y, p))

show_fig(test_x, test_y, LR.w, posibility)
