import csv
import numpy as np
from linear_regression import LinearRegressionMatrix
from functools import partial
from softmax import Softmax
from utils import show_lineChart

# dataset
data = []
n_class = 0
with open('iris/iris.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader) # skip header
    class_name = None
    for row in reader:
        if row[5] != class_name:
            # print(row[5], class_name)
            class_name = row[5]
            n_class += 1
        data.append(list(map(float, row[1:5])) + [n_class])

data = np.array(data)
data = data.reshape(3, 50, 5)
train_set = []
test_set = []
for i in range(data.shape[0]):
    np.random.shuffle(data[i])
    train_set.append(data[i][:30])
    test_set.append(data[i][30:])

# OVO
OVO_classifiers = []
for i in range(n_class - 1):
    OVO_classifiers.append([])
    for j in range(i+1, n_class):
        x = np.concatenate((train_set[i][:, :4], train_set[j][:, :4]), axis=0)
        y = np.concatenate((np.ones((train_set[i].shape[0], 1)), 
                            -np.ones((train_set[j].shape[0], 1))), axis=0)
        LR = LinearRegressionMatrix(x, y)
        OVO_classifiers[i].append(LR)

x = np.concatenate(test_set, axis=0)[:, :4]
y = np.concatenate(test_set, axis=0)[:, 4]
y_predict = []
for i in range(n_class - 1):
    for j in range(i+1, n_class):
        sign_pred_res = np.sign(OVO_classifiers[i][j-i-1].predict(x))
        pred_res = np.zeros_like(sign_pred_res)
        pred_res[sign_pred_res == 1] = i + 1
        pred_res[sign_pred_res == -1] = j + 1
        y_predict.append(pred_res[:, None])
y_predict = np.concatenate(y_predict, axis=1)

# vote
u_c = np.apply_along_axis(partial(np.unique, return_counts=True), axis=1, arr=y_predict).squeeze()
unique = u_c[:, 0]
counts = u_c[:, 1]
most_common_index = np.argmax(counts, axis=1)
y_hat = unique[np.arange(most_common_index.shape[0]), most_common_index]

print(np.equal(y_hat, y).sum() / y.shape[0])

# softmax
x = np.concatenate(train_set, axis=0)[:, :4]
y = np.concatenate(train_set, axis=0)[:, 4]
y = np.eye(n_class)[y.astype(int) - 1] # 从单位矩阵中取出标签对应行拼起来
softmax = Softmax(x, y)
loss_list, acc_list = softmax.train(500, 0.05, batch_size=90)

x = np.concatenate(test_set, axis=0)[:, :4]
y = np.concatenate(test_set, axis=0)[:, 4]
show_lineChart(loss_list, xlabel='epoch', ylabel='loss')
show_lineChart(acc_list, xlabel='epoch', ylabel='accuracy')
y_hat = np.argmax(softmax(x), axis=1) + 1

print(np.equal(y_hat, y).sum() / y.shape[0])