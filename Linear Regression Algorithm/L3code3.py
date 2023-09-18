import numpy as np
from utils import gen_two_random_normal, show_fig
from linear_regression import LinearRegressionMatrix, LinearRegression
from optimizer import GradientDescent

# 生成数据集
m1 = np.array([1, 0])
s1 = np.matrix([[1, 0], [0, 1]])
size1 = 200
m2 = np.array([0, 1])
s2 = np.matrix([[1, 0], [0, 1]])
size2 = 200

x_train, y_train, x_test, y_test = gen_two_random_normal(m1, s1, size1, m2, s2, size2, 
                                                         training_set_ratio=0.8)

show_fig(x_train, y_train)

# 最小二乘法
LRM = LinearRegressionMatrix(x_train, y_train)
show_fig(x_train, y_train, LRM.w)
show_fig(x_test, y_test, LRM.w)

optim = GradientDescent(lr=0.01)
LRG = LinearRegression(optim, x_train, y_train)
LRG.train(150, show_loss=True)
show_fig(x_train, y_train, LRG.w)
show_fig(x_test, y_test, LRG.w)

print('least squares w: {}\ngradient descent w: {}'.format(LRM.w, LRG.w))

# 统计正确率

LRM_train_y = np.sign(LRM.predict(x_train))
LRM_test_y = np.sign(LRM.predict(x_test))
LRM_train_accuracy = np.sum(LRM_train_y == y_train) / y_train.shape[0]
LRM_test_accuracy = np.sum(LRM_test_y == y_test) / y_test.shape[0]

LRG_train_y = np.sign(LRG.predict(x_train))
LRG_test_y = np.sign(LRG.predict(x_test))
LRG_train_accuracy = np.sum(LRG_train_y == y_train) / y_train.shape[0]
LRG_test_accuracy = np.sum(LRG_test_y == y_test) / y_test.shape[0]

print('least squares train accuracy: {}\nleast squares test accuracy: {}'
      .format(LRM_train_accuracy, LRM_test_accuracy))

print('gradient descent train accuracy: {}\ngradient descent test accuracy: {}'
      .format(LRG_train_accuracy, LRG_test_accuracy))
