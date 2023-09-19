from utils import show_fig, gen_two_random_normal, get_accuracy
from Fisher import Fisher
import numpy as np

m1 = [-5, 0]
s1 = [[1, 0], [0, 1]]
size1 = 200
m2 = [0, 5]
s2 = [[1, 0], [0, 1]]
size2 = 200
train_x, train_y, test_x, test_y = gen_two_random_normal(m1, s1, size1, m2, s2, size2)

# import pdb; pdb.set_trace()
fisher = Fisher(train_x[np.squeeze(np.array(train_y), axis=1)==1], 
                train_x[np.squeeze(np.array(train_y), axis=1)==-1])

y_predict = fisher.predict(test_x)
show_fig(test_x, test_y, y_predict, fisher.w, fisher.threshold)

print('weight: \n{}\n threshold:{}'.format(fisher.w, fisher.threshold))

get_accuracy('Fisher', fisher, train_x, train_y, test_x, test_y)
