import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Callable, Union
from functools import partial
import os

# SVM的增广样本b在x的第一个，也就是x[0]，所以这里的画图有一定区别

# def show_fig(x, 
#              y, 
#              w: Optional[np.ndarray]=None,
#              pred_func: Optional[Callable]=None,
#              predictor: Optional[Union[Dual_SVM, Kernel_SVM]]=None,
#              save_path: Optional[str]=None,
#             ):
#     draw_points(x, y)
#     if w is not None:
#         draw_line(x, w, 'b-')
#     if pred_func is not None:
#         plot_decision_boundary(x, pred_func)
#     plt.axis('equal')
#     xmin = x[:, 0].min()
#     xmax = x[:, 0].max()
#     ymin = x[:, 1].min()
#     ymax = x[:, 1].max()
#     padding = (xmax - xmin) * 0.2
#     plt.xlim(xmin - padding, xmax + padding)
#     plt.ylim(ymin - padding, ymax + padding)

#     if predictor is not None:
#         draw_boundary(predictor, xmin, xmax, x, pred_func)

#     plt.show(block=False)

#     if save_path is not None:
#         if not os.path.exists(os.path.dirname(save_path)):
#             os.mkdir(os.path.dirname(save_path))
#         plt.savefig(save_path, dpi=300)
#     plt.waitforbuttonpress()
#     plt.close()

def draw_points(x: np.ndarray, 
                y: np.ndarray):
    plt.scatter(*(x[np.squeeze(y==1)].T), color='g', s=15)
    plt.scatter(*(x[np.squeeze(y==-1)].T), color='r', s=15)

def draw_line(x: np.ndarray,
              w: np.ndarray,
              cfg):
    xmin = x[:, 0].min()
    xmax = x[:, 0].max()
    if w[1] != 0:
        a = -w[1] / w[2]
        b = -w[0] / w[2]
    else:
        a = 0
        b = 0
    x = np.array([xmin, xmax])
    y = a * x + b
    plt.plot(x, y, cfg)

# def draw_boundary(predictor: Union[np.ndarray, Dual_SVM, Kernel_SVM] = None,
#                   xmin=None,
#                   xmax=None,
#                   x=None,
#                   pred_func=None
#                  ):
#     if isinstance(predictor, Dual_SVM):
#         assert(xmin is not None and xmax is not None)
#         pos_x = predictor.sv_x[predictor.sv_y.squeeze() == 1]
#         neg_x = predictor.sv_x[predictor.sv_y.squeeze() == -1]
#         plt.scatter(*(pos_x.T), color="darkgreen", s=40)
#         plt.scatter(*(neg_x.T), color="darkred", s=40)
#         x = np.array([xmin, xmax])
#         y = (x - pos_x[0, 0]) * (-predictor.w[1] / predictor.w[2]) + pos_x[0, 1]
#         plt.plot(x, y, 'g--')
#         y = (x - neg_x[0, 0]) * (-predictor.w[1] / predictor.w[2]) + neg_x[0, 1]
#         plt.plot(x, y, 'r--')
#     elif isinstance(predictor, Kernel_SVM):
#         assert(x is not None and pred_func is not None)
#         pos_x = predictor.sv_x[predictor.sv_y.squeeze() == 1]
#         neg_x = predictor.sv_x[predictor.sv_y.squeeze() == -1]

#         pos_b = -np.multiply(predictor.sv_alpha * predictor.sv_y, 
#                              predictor.kernel_func(predictor.sv_x, 
#                                                    np.expand_dims(pos_x[0], axis=0))
#                             ).sum(axis=0)

#         neg_b = -np.multiply(predictor.sv_alpha * predictor.sv_y,
#                              predictor.kernel_func(predictor.sv_x,
#                                                    np.expand_dims(neg_x[0], axis=0))
#                             ).sum(axis=0)
        
#         plt.scatter(*(pos_x.T), color="darkgreen", s=40)
#         plt.scatter(*(neg_x.T), color="darkred", s=40)
#         plot_decision_boundary(x, pred_func=partial(pred_func, b=pos_b), h=0.05)
#         plot_decision_boundary(x, pred_func=partial(pred_func, b=neg_b), h=0.05)

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
    predict_train_y = predicter(x_train)
    predict_test_y = predicter(x_test)
    train_accuracy = np.sum(predict_train_y == y_train) / y_train.shape[0]
    test_accuracy = np.sum(predict_test_y == y_test) / y_test.shape[0]

    print('{} train accuracy: {}\n{} test accuracy: {}'
        .format(name, train_accuracy, name, test_accuracy))
    
def plot_decision_boundary(x, pred_func, h=0.05):  
    x_min = x[:, 0].min()
    x_max = x[:, 0].max()
    y_min = x[:, 1].min()
    y_max = x[:, 1].max()
    padding = (x_max - x_min) * 0.2
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)) 

    # 预测 
    pred_label = pred_func(np.c_[xx.ravel(), yy.ravel()])  
    pred_label = pred_label.reshape(xx.shape)

    plt.contourf(xx, yy, pred_label, cmap='RdYlGn', alpha=0.2)
