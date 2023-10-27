import numpy as np

'''
核函数功能
根据输入x1, x2，计算核函数值

若x1, x2包含多个样本
x1: (n, d)
x2: (m, d)
计算结果为: (n, m)
结果的每个元素为: K(x1[n], x2[m])
'''

def gauss_kernel(x1: np.ndarray, 
                 x2: np.ndarray, 
                 gamma: float):
    '''
    x1: (n, d)
    x2: (m, d)
    计算结果为: (n, m)
    结果的每个元素为: K(x1[n], x2[m])
    '''
    n, dim1 = x1.shape
    m, dim2 = x2.shape
    assert dim1 == dim2, "x1 and x2 must have the same dimension"

    # import pdb; pdb.set_trace()

    # 把x1扩展为(n, m, d)
    x1 = np.tile(x1[None, :, :], (m, 1, 1))
    x1 = x1.transpose((1, 0, 2))

    # 把x2扩展为(n, m, d)
    x2 = np.tile(x2[None, :, :], (n, 1, 1))

    kernel_value = np.exp(-gamma * np.linalg.norm(x1 - x2, axis=2) ** 2)
    
    return kernel_value

def polynomial_kernel(x1, x2, zeta, gamma, order):
    '''
    x1: (n, d)
    x2: (m, d)
    计算结果为: (n, m)
    结果的每个元素为: K(x1[n], x2[m])
    '''
    _, dim1 = x1.shape
    _, dim2 = x2.shape
    assert dim1 == dim2, "x1 and x2 must have the same dimension"

    kernel_value = (zeta + gamma * x1 @ x2.T) ** order
    return kernel_value

if __name__ == "__main__":
    x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x2 = np.array([[1, 2, 3], [4, 5, 6]])
    gauss_kernel(x1, x2, 1)
    print(polynomial_kernel(x1, x1, 1, 1, 2))
