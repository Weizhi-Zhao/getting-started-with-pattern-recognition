import numpy as np
from utils import show_fig, draw_line, draw_points
import matplotlib.pyplot as plt

class PLA:
    def __init__(self, inD,
                 x: np.ndarray, 
                 y: np.ndarray,
                 show_steps=True) -> None:
        # 特征维度
        self.inD = inD
        # 权向量（增广）
        self.w = np.zeros(inD + 1)
        bias = np.ones(x.shape[0])
        # 特征向量增广
        self.x = np.insert(x, x.shape[1], values=bias, axis=1)
        self.y = y
        # 是否显示每一步的结果
        self.show_steps = show_steps

    # xn: 一个样本，x: 所有样本
    def update(self, 
               xn: np.ndarray, 
               yn: float) -> None:
        self.w = self.w + yn * xn

    def predict(self, 
                xn: np.ndarray) -> float:
        return np.sign(self.w @ xn.T)
    
    def train(self) -> None:
        while True:
            for xn, yn in zip(self.x, self.y):
                if self.predict(xn) != yn:
                    self.update(xn, yn)
                    if self.show_steps == True:
                        self.show_step_result()
                        print("w: [{}; {}; {}]".format(self.w[0], self.w[1], self.w[2]))
                    break
            else: # for循环的else在完整的循环结束后执行
                break

    def show_step_result(self):
        # 注意，此时的x已经增广过了
        draw_line(self.x[:, 0:2], self.y, self.w, 'b-')
        draw_points(self.x[:, 0:2], self.y)
        plt.axis('equal')
        xmin = self.x[:, 0].min()
        xmax = self.x[:, 0].max()
        ymin = self.x[:, 1].min()
        ymax = self.x[:, 1].max()
        padding = (xmax - xmin) * 0.2
        plt.xlim(xmin - padding, xmax + padding)
        plt.ylim(ymin - padding, ymax + padding)
        plt.draw()
        plt.waitforbuttonpress()
        plt.cla()