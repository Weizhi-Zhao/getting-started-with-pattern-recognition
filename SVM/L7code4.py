from utils import gen_two_random_normal, show_fig, get_accuracy
from SVMs import SVM, Dual_SVM, Kernel_SVM
from functools import partial
from kernel import gauss_kernel, polynomial_kernel
import matplotlib.pyplot as plt
import numpy as np

save_path = "code4 results/"

m1 = [2, 0]
s1 = [[2, 0], [0, 2]]
size1 = 200
m2 = [0, 2]
s2 = [[2, 0], [0, 2]]
size2 = 200
train_x, train_y, test_x, test_y = gen_two_random_normal(m1, s1, size1, m2, s2, size2)
'''
# Primal SVM
primal_svm = SVM(train_x, train_y)
primal_svm.quadratic_programming()
show_fig(test_x, test_y, w=primal_svm.w, save_path=save_path+'primal_svm.png')
get_accuracy('primal_svm', lambda x: np.sign(SVM.augment(x) @ primal_svm.w)[:, None], 
             train_x, train_y, test_x, test_y)
'''
'''
# Dual SVM
dual_svm = Dual_SVM(train_x, train_y)
dual_svm.quadratic_programming()
show_fig(test_x, test_y, w=dual_svm.w, save_path=save_path+'dual_svm.png')
get_accuracy('dual_svm', lambda x: np.sign(SVM.augment(x) @ dual_svm.w)[:, None], 
             train_x, train_y, test_x, test_y)
show_fig(train_x, train_y, w=dual_svm.w, predictor=dual_svm, save_path=save_path+'Dsvm Bline.png')
'''

# Kernel SVM
poly_k = partial(polynomial_kernel, zeta=2, gamma=1, order=4)
gauss_k = partial(gauss_kernel, gamma=0.5)
poly_kernel_svm = Kernel_SVM(train_x, train_y, poly_k)
gauss_kernel_svm = Kernel_SVM(train_x, train_y, gauss_k)
poly_kernel_svm.quadratic_programming()
gauss_kernel_svm.quadratic_programming()
show_fig(test_x, test_y, pred_func=poly_kernel_svm, save_path=save_path+'poly_kernel_svm.png')
show_fig(train_x, train_y, pred_func=poly_kernel_svm, predictor=poly_kernel_svm, 
         save_path=save_path+'poly Bline.png')
show_fig(test_x, test_y, pred_func=gauss_kernel_svm, save_path=save_path+'gauss_kernel_svm.png')
show_fig(train_x, train_y, pred_func=gauss_kernel_svm, predictor=gauss_kernel_svm,
         save_path=save_path+'gauss Bline.png')
get_accuracy('polynomial_kernel_svm', poly_kernel_svm, train_x, train_y, test_x, test_y)
get_accuracy('gauss_kernel_svm', gauss_kernel_svm, train_x, train_y, test_x, test_y)

with open(save_path+"train and test set.txt", '+wt') as f:
    # f.write("support vector of Dual SVM:\n")
    # f.write("x:\n")
    # f.write(str(dual_svm.sv_x))
    # f.write("\ny:\n")
    # f.write(str(dual_svm.sv_y))
    # f.write("\nalpha:\n")
    # f.write(str(dual_svm.sv_alpha))

    f.write("\n\nsupport vrctor of Polynomial Kernel SVM:\n")
    f.write("x:\n")
    f.write(str(poly_kernel_svm.sv_x))
    f.write("\ny:\n")
    f.write(str(poly_kernel_svm.sv_y))
    f.write("\nalpha:\n")
    f.write(str(poly_kernel_svm.sv_alpha))

    f.write("\n\nsupport vrctor of Gaussian Kernel SVM:\n")
    f.write("x:\n")
    f.write(str(gauss_kernel_svm.sv_x))
    f.write("\ny:\n")
    f.write(str(gauss_kernel_svm.sv_y))
    f.write("\nalpha:\n")
    f.write(str(gauss_kernel_svm.sv_alpha))

    f.write('\n\ntrain_x:\n')
    f.write(str(train_x))
    f.write('\ntrain_y:\n')
    f.write(str(train_y))
    f.write('\ntest_x:\n')
    f.write(str(test_x))
    f.write('\ntest_y:\n')
    f.write(str(test_y))
