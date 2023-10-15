import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def show_fig(x: np.ndarray, 
             y: np.ndarray, 
             w=None,
             possibility: Optional[np.ndarray] = None):
    if possibility is not None:
        draw_points(x, y, possibility)
    else:
        draw_points(x, y)
    if w is not None:
        draw_line(x, w, 'b-')

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
                y: np.ndarray,
                possibility: Optional[np.ndarray] = None):
    if possibility is not None:
        possibility = (possibility - possibility.min()) / (possibility.max() - possibility.min())
    for i in range(x.shape[0]):
        xn = x[i]
        yn = y[i]
        if yn == 1:
            if possibility is not None:
                plt.plot(*(xn), marker='o', color=(0, possibility[i], 0))
            else:
                plt.plot(*(xn), 'g.')
        else:
            if possibility is not None:
                plt.plot(*(xn), marker='o', color=(possibility[i], 0, 0))
            else:
                plt.plot(*(xn), 'r.')

def draw_line(x: np.ndarray, 
              w: np.ndarray,
              cfg):
    xmin = x[:, 0].min()
    xmax = x[:, 0].max()
    if w[2] != 0:
        a = -w[1] / w[2]
        b = -w[0] / w[2]
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
