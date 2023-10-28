from SVMs import SVM, Dual_SVM, Kernel_SVM
from utils import show_fig
import numpy as np
import matplotlib.pyplot as plt
from kernel import gauss_kernel
from functools import partial

save_path = "code5 results/"

# 钓鱼岛坐标
钓鱼岛 = [123.28,25.45]
钓鱼岛 = np.array(钓鱼岛)[None, :]

# 仅仅使用沿海城市

China_coastal = [119.28,26.08,#福州
                 121.31,25.03,#台北
                 121.47,31.23,#上海
                 118.06,24.27,#厦门
                 121.46,39.04,#大连
                 122.10,37.50,#威海
                 124.23,40.07]#丹东

Japan_coastal = [129.87,32.75,#长崎
                 130.33,31.36,#鹿儿岛
                 131.42,31.91,#宫崎
                 130.24,33.35,#福冈
                 133.33,15.43,#鸟取
                 138.38,34.98,#静冈
                 140.47,36.37]#水户  

# 添加内陆城市

China = [119.28,26.08,#福州
         121.31,25.03,#台北
         121.47,31.23,#上海
         118.06,24.27,#厦门
         113.53,29.58,#武汉
         104.06,30.67,#成都
         116.25,39.54,#北京
         121.46,39.04,#大连
         122.10,37.50,#威海
         124.23,40.07]#丹东

Japan = [129.87,32.75,#长崎
         130.33,31.36,#鹿儿岛
         131.42,31.91,#宫崎
         130.24,33.35,#福冈
         136.54,35.10,#名古屋
         132.27,34.24,#广岛
         139.46,35.42,#东京
         133.33,15.43,#鸟取
         138.38,34.98,#静冈
         140.47,36.37]#水户

# 只看沿海城市的情况
x = np.concatenate((China_coastal, Japan_coastal), axis=0).reshape(-1, 2)
y = np.concatenate((np.ones((len(China_coastal)//2, 1)), 
                    -np.ones((len(Japan_coastal)//2, 1))), axis=0)

dual_svm = Dual_SVM(x, y)
dual_svm.quadratic_programming()
plt.scatter(*(钓鱼岛.squeeze()), color='y', marker='x', s=100)
show_fig(x, y, w=dual_svm.w, predictor=dual_svm, save_path=save_path+'Dsvm sea.png')
print(np.sign(SVM.augment(钓鱼岛) @ dual_svm.w))

gauss_k = partial(gauss_kernel, gamma=0.1)
kernel_svm = Kernel_SVM(x, y, gauss_k)
kernel_svm.quadratic_programming()
plt.scatter(*(钓鱼岛.squeeze()), color='y', marker='x', s=100)
show_fig(x, y, pred_func=kernel_svm, save_path=save_path+'Ksvm sea.png')
# show_fig(x, y, pred_func=kernel_svm, predictor=kernel_svm, save_path=save_path+'Ksvm sea.png')
print(kernel_svm(钓鱼岛))

# 加入内陆城市
x = np.concatenate((China, Japan), axis=0).reshape(-1, 2)
y = np.concatenate((np.ones((len(China)//2, 1)), 
                    -np.ones((len(Japan)//2, 1))), axis=0)

dual_svm = Dual_SVM(x, y)
dual_svm.quadratic_programming()
plt.scatter(*(钓鱼岛.squeeze()), color='y', marker='x', s=100)
show_fig(x, y, w=dual_svm.w, predictor=dual_svm, save_path=save_path+'Dsvm land.png')
print(np.sign(SVM.augment(钓鱼岛) @ dual_svm.w))

gauss_k = partial(gauss_kernel, gamma=0.1)
kernel_svm = Kernel_SVM(x, y, gauss_k)
kernel_svm.quadratic_programming()
plt.scatter(*(钓鱼岛.squeeze()), color='y', marker='x', s=100)
show_fig(x, y, pred_func=kernel_svm, save_path=save_path+'Ksvm land.png')
# show_fig(x, y, pred_func=kernel_svm, predictor=kernel_svm, save_path=save_path+'Ksvm land.png')
print(kernel_svm(钓鱼岛))