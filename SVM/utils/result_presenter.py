import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# SVM的增广样本b在x的第一个，也就是x[0]，所以这里的画图有一定区别

def show_fig(x, 
             y, 
             w=None):
    draw_points(x, y)
    if w is not None:
        draw_line(x, y, w, 'b-')
    plt.axis('equal')
    xmin = x[:, 0].min()
    xmax = x[:, 0].max()
    ymin = x[:, 1].min()
    ymax = x[:, 1].max()
    padding = (xmax - xmin) * 0.2
    plt.xlim(xmin - padding, xmax + padding)
    plt.ylim(ymin - padding, ymax + padding)
    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

def draw_points(x: np.ndarray, 
                y: np.ndarray):
    y = np.array(y)
    x = np.array(x)
    for xn, yn in zip(x, y):
        if yn == 1:
            plt.plot(*(xn), 'g.')
        else:
            plt.plot(*(xn), 'r.')

def draw_line(x: np.matrix, 
              y: np.matrix, 
              w: np.matrix,
              cfg):
    xmin = x[:, 0].min()
    xmax = x[:, 0].max()
    if w[1, 0] != 0:
        a = -w[1, 0] / w[2, 0]
        b = -w[0, 0] / w[2, 0]
    else:
        a = 0
        b = 0
    x = np.arange(xmin, xmax, 0.01)
    y = a * x + b
    plt.plot(x, y, cfg)

def show_lineChart(values: list,
                   xlabel: Optional[str] = None,
                   ylabel: Optional[str] = None):
    plt.plot(values)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    plt.show(block=False)
    plt.waitforbuttonpress()
    plt.close()

def get_accuracy(name, predicter, x_train, y_train, x_test, y_test):
    predict_train_y = np.sign(predicter.predict(x_train))
    predict_test_y = np.sign(predicter.predict(x_test))
    train_accuracy = np.sum(predict_train_y == y_train) / y_train.shape[0]
    test_accuracy = np.sum(predict_test_y == y_test) / y_test.shape[0]

    print('{} train accuracy: {}\n{} test accuracy: {}'
        .format(name, train_accuracy, name, test_accuracy))
