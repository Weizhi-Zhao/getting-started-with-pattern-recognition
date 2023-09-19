import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def show_fig(x: np.matrix, 
             y: np.matrix, 
             y_predict: np.matrix,
             w=None,
             threshold=None):
    draw_predict_points(x, y_predict)
    draw_points(x, y)
    if w is not None:
        draw_projection_line(x, w, threshold)

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

def draw_points(x: np.matrix, 
                y: np.matrix):
    x = np.array(x)
    for xn, yn in zip(x, y):
        if yn == 1:
            plt.plot(*(xn), 'g.')
        else:
            plt.plot(*(xn), 'r.')

def draw_predict_points(x: np.matrix, 
                        y_predict: np.matrix):
    x = np.array(x)
    for xn, yn in zip(x, y_predict):
        if yn == 1:
            plt.plot(*(xn), 'g.', markersize=11)
        else:
            plt.plot(*(xn), 'r.', markersize=11)

def draw_projection_line(x: np.matrix,
              w: np.matrix,
              threshold):
    xmin = x[:, 0].min()
    xmax = x[:, 0].max()
    ymin = x[:, 1].min()
    ymax = x[:, 1].max()

    mid_x = (xmax + xmin) / 2
    mid_y = (ymax + ymin) / 2

    if w[0, 0] != 0:
        a = w[1, 0] / w[0, 0]
    else:
        a = 0
    x = np.arange(xmin, xmax, 0.01)
    y = a * (x - mid_x) + mid_y
    plt.plot(x, y, 'b-')

    x = (threshold + a * w[1, 0] * mid_x - w[1, 0] * mid_y) / (w[0, 0] + a * w[1, 0])
    y = a * (x - mid_x) + mid_y
    plt.plot(x, y, 'y*', markersize=15)

def get_accuracy(name, predicter, x_train, y_train, x_test, y_test):
    predict_train_y = np.sign(predicter.predict(x_train))
    predict_test_y = np.sign(predicter.predict(x_test))
    train_accuracy = np.sum(predict_train_y == y_train) / y_train.shape[0]
    test_accuracy = np.sum(predict_test_y == y_test) / y_test.shape[0]

    print('{} train accuracy: {}\n{} test accuracy: {}'
        .format(name, train_accuracy, name, test_accuracy))
