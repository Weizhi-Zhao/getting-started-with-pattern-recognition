import numpy as np
import matplotlib.pyplot as plt
from utils import gen_two_random_normal, show_fig
from PLA import PLA

m1 = [-5, 0]
cov1 = [[1, 0], [0, 1]]
size1 = 200
m2 = [0, 5]
cov2 = [[1, 0], [0, 1]]
size2 = 200
training_x, training_y, test_x, test_y = gen_two_random_normal(m1, cov1, size1, m2, cov2, size2, 0.8)

show_fig(training_x, training_y)
show_fig(test_x, test_y)

pla = PLA(2, training_x, training_y, show_steps=True)
pla.train()

show_fig(test_x, test_y, w=pla.w)
