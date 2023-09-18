import matplotlib.pyplot as plt
import numpy as np

def show_fig(x: np.ndarray, 
             y: np.ndarray, 
             y_predict: np.ndarray):
    draw_predict_points(x, y_predict)
    draw_points(x, y)

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
    for xn, yn in zip(x, y):
        if yn == 1:
            plt.plot(*(xn), 'g.')
        else:
            plt.plot(*(xn), 'r.')

def draw_predict_points(x: np.ndarray, 
                        y_predict: np.ndarray):
    for xn, yn in zip(x, y_predict):
        if yn == 1:
            plt.plot(*(xn), 'b^', markersize=10)
        else:
            plt.plot(*(xn), 'y^', markersize=10)

