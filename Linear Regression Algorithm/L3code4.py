import numpy as np
from utils import gen_two_random_normal, show_fig, get_accuracy
from linear_regression import LinearRegressionMatrix, LinearRegression, LinearRegressionSDG
from optimizer import GradientDescent, Adagrad

# 生成数据集
m1 = np.array([2, 0])
s1 = np.matrix([[5, 0], [0, 5]])
size1 = 200
m2 = np.array([0, -2])
s2 = np.matrix([[5, 0], [0, 5]])
size2 = 200

x_train, y_train, x_test, y_test = gen_two_random_normal(m1, s1, size1, m2, s2, size2, 
                                                         training_set_ratio=0.8)

show_fig(x_train, y_train)

# 最小二乘法
LRM = LinearRegressionMatrix(x_train, y_train)
show_fig(x_train, y_train, LRM.w)
show_fig(x_test, y_test, LRM.w)

# 梯度下降
optim = GradientDescent(lr=0.1)
LR_GD = LinearRegression(optim, x_train, y_train)
LR_GD.train(100, show_loss=True)
show_fig(x_train, y_train, LR_GD.w)
show_fig(x_test, y_test, LR_GD.w)

# 随机梯度下降
optim = GradientDescent(lr=0.1)
LRSGD = LinearRegressionSDG(optim, x_train, y_train, batch_size=20)
LRSGD.train(50, show_loss=True)
show_fig(x_train, y_train, LRSGD.w)
show_fig(x_test, y_test, LRSGD.w)

print('least squares w: {}\nSGD w:{}\ngradient descent w: {}'.format(LRM.w, LR_GD.w, LRSGD.w))

# 统计正确率
get_accuracy('least squares', LRM, x_train, y_train, x_test, y_test)
get_accuracy('gridient descent', LR_GD, x_train, y_train, x_test, y_test)
get_accuracy('SGD', LRSGD, x_train, y_train, x_test, y_test)
