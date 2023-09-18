import numpy as np
import matplotlib.pyplot as plt
from utils import gen_two_random_normal, show_fig
from PLA import PLA
from pocket import Pocket
from timeit import timeit

m1 = [1, 0]
cov1 = [[1, 0], [0, 1]]
size1 = 200
m2 = [0, 1]
cov2 = [[1, 0], [0, 1]]
size2 = 200

# 自己编写的生成训练集、测试集函数
training_x, training_y, test_x, test_y = gen_two_random_normal(m1, cov1, size1, m2, cov2, size2, 0.8)

# 用自己编写的绘图函数，展示训练集和测试集数据
show_fig(training_x, training_y)
show_fig(test_x, test_y)

# 训练PLA和Pocket算法，并统计时间
# pla = PLA(training_x, training_y, show_steps=False)
# print('PLA training time: {}'.format(timeit('pla.train()', number=1, globals=globals())))
pocket = Pocket(training_x, training_y, iter_num=100, show_steps=False)
print('pocket training time: {}'.format(timeit('pocket.train()', number=1, globals=globals())))

# 打印分类面参数
# print('PLA w: {}'.format(pla.w))
print('Pocket w: {}'.format(pocket.pocket['weight']))

# 使用PLA和pocket算法进行预测
augmented_train_x = np.insert(training_x, training_x.shape[1], values=np.ones(training_x.shape[0]), axis=1)
augmented_test_x = np.insert(test_x, test_x.shape[1], values=np.ones(test_x.shape[0]), axis=1)

# pla_train_y = pla.predict(augmented_train_x)
# pla_test_y = pla.predict(augmented_test_x)

pocket_train_y = pocket.predict(augmented_train_x)
pocket_test_y = pocket.predict_with_pocket(augmented_test_x)

# 统计PLA和Pocket算法在训练集和测试集上的正确率

# PLA_train_accuracy = np.sum(pla_train_y == training_y.T) / training_y.shape[0]
# PLA_test_accuracy = np.sum(pla_test_y == test_y.T) / test_y.shape[0]
Pocket_train_accuracy = np.sum(pocket_train_y == training_y.T) / training_y.shape[0]
Pocket_test_accuracy = np.sum(pocket_test_y == test_y.T) / test_y.shape[0]

# print('PLA train accuracy: {}'.format(PLA_train_accuracy))
# print('PLA test accuracy: {}'.format(PLA_test_accuracy))
print('Pocket train accuracy: {}'.format(Pocket_train_accuracy))
print('Pocket test accuracy: {}'.format(Pocket_test_accuracy))

# show_fig(test_x, test_y, w=pla.w)
show_fig(test_x, test_y, w=pocket.pocket['weight'])
