import numpy as np

# 生成由两个随机正态分布的二维向量组成的数据集
def gen_two_random_normal(mean_p, cov_p, size_p, mean_n, cov_n, size_n, training_set_ratio=0.8):
    x_p = np.random.multivariate_normal(mean_p, cov_p, size_p)
    y_p = np.ones(size_p)

    x_n = np.random.multivariate_normal(mean_n, cov_n, size_n)
    y_n = -np.ones(size_n)

    x = np.concatenate((x_p, x_n), axis=0)
    y = np.concatenate((y_p, y_n), axis=0)

    data = np.concatenate((x, y[:, None]), axis=1)
    np.random.shuffle(data)

    training_set, test_set = np.split(data, [int(len(data) * training_set_ratio)])

    training_x, training_y = np.split(training_set, [2], axis=1)
    test_x, test_y = np.split(test_set, [2], axis=1)

    return training_x, training_y, test_x, test_y
