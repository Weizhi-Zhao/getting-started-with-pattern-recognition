import numpy as np
import random
from utils import draw_points, draw_line
import matplotlib.pyplot as plt
import os

# pocket算法，解决线性不可分情况
class Pocket:
    def __init__(self, inD,
                 x: np.ndarray, 
                 y: np.ndarray,
                 iter_num, show_steps=True, save_fig=False) -> None:
        # 特征维度
        self.inD = inD
        # 权向量（增广）
        self.w = np.zeros(inD + 1)
        self.pocket = {'weight': self.w,
                       'correct': 0}
        bias = np.ones(x.shape[0])
        # 特征向量增广
        self.x = np.insert(x, x.shape[1], values=bias, axis=1)
        self.y = y
        # 迭代次数
        self.iter_num = iter_num
        # 是否显示每一步的结果
        self.show_steps = show_steps
        # 是否保存过程中的结果图片（只有show_steps==True时起作用）
        self.save_fig = save_fig

    # xn: 一个样本，x: 所有样本
    def update(self, 
               xn: np.ndarray, 
               yn: float) -> None:
        self.w = self.w + yn * xn

    # 用当前权向量预测
    def predict(self, 
                xn: np.ndarray) -> int:
        return np.sign(self.w @ xn.T)
    
    def train(self) -> None:
        for i in range(self.iter_num):
            # 记录所有分类错误的样本
            error_samples = []
            # 统计分类正确的次数
            correct = 0
            for xn, yn in zip(self.x, self.y):
                if self.predict(xn) != yn:
                    error_samples.append({'xn': xn, 'yn': yn})
                else:
                    correct += 1
                    plt.plot(xn[0], xn[1], 'yv', markersize=10)

            if self.show_steps == True:
                self.show_step_result(i)
                print('t = {}'.format(i))
                print("now w         : [{}; {}; {}]".format(self.w[0], self.w[1], self.w[2]))
                print("pocket w      : [{}; {}; {}]".format(self.pocket['weight'][0], 
                                                            self.pocket['weight'][1], 
                                                            self.pocket['weight'][2]))
                print("now correct   : {}".format(correct))
                print("pocket correct: {}".format(self.pocket['correct']))
                print('\n')
            
            # 如果当前权重更优，更新pocket
            if correct > self.pocket['correct']:
                self.pocket['weight'] = self.w
                self.pocket['correct'] = correct

            if len(error_samples) == 0:
                break
            
            # 随机选择一个错误分类的样本，用于更新权向量
            rand_err_sample = random.choice(error_samples)
            self.update(rand_err_sample['xn'], rand_err_sample['yn'])

    def show_step_result(self, step_num):
        # 注意，此时的x已经增广过了
        draw_line(self.x[:, 0:2], self.y, self.w, 'b-')
        draw_line(self.x[:, 0:2], self.y, self.pocket['weight'], 'g-')
        draw_points(self.x[:, 0:2], self.y)
        plt.axis('equal')
        xmin = self.x[:, 0].min()
        xmax = self.x[:, 0].max()
        ymin = self.x[:, 1].min()
        ymax = self.x[:, 1].max()
        padding = (xmax - xmin) * 0.2
        plt.xlim(xmin - padding, xmax + padding)
        plt.ylim(ymin - padding, ymax + padding)
        if self.save_fig == True:
            if not os.path.exists('figs'):
                os.makedirs('figs')
            plt.savefig(os.path.join('figs', 'fig{}.png'.format(step_num)), dpi=300)
        plt.draw()
        # plt.waitforbuttonpress()
        plt.pause(0.3)
        plt.cla()

    # 依据最优权向量预测
    def predict_with_pocket(self, 
                xn: np.ndarray) -> float:
        return np.sign(self.pocket['weight'] @ xn.T)