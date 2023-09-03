import matplotlib.pyplot as plt
import numpy as np

def show_fig(x: np.ndarray, 
             y: np.ndarray, 
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
    for xn, yn in zip(x, y):
        if yn == 1:
            plt.plot(*(xn.T), 'g.')
        else:
            plt.plot(*(xn.T), 'r.')

def draw_line(x: np.ndarray, 
              y: np.ndarray, 
              w: np.ndarray,
              cfg):
    xmin = x[:, 0].min()
    xmax = x[:, 0].max()
    if w[1] != 0:
        a = -w[0] / w[1]
        b = -w[2] / w[1]
    else:
        a = 0
        b = 0
    x = np.arange(xmin, xmax, 0.01)
    y = a * x + b
    plt.plot(x, y, cfg)
