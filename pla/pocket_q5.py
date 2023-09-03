import numpy as np
import matplotlib.pyplot as plt
from pocket import Pocket
from utils import show_fig

training_x = np.array(
    [
        [0.2, 0.7],
        [0.3, 0.3],
        [0.4, 0.5],
        [0.6, 0.5],
        [0.1, 0.4],
        [0.4, 0.6],
        [0.6, 0.2],
        [0.7, 0.4],
        [0.8, 0.6],
        [0.7, 0.5]
    ]
)
training_y = np.array(
    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1]
)

show_fig(training_x, training_y)

pocket = Pocket(2, training_x, training_y, iter_num=50, show_steps=True, save_fig=False)
pocket.train()

show_fig(training_x, training_y, pocket.pocket['weight'])
