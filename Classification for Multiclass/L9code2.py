from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from softmax import Softmax
from utils import show_lineChart
import matplotlib.pyplot as plt

# dataset
train_set = MNIST(root='mnist', train=True, transform=transforms.ToTensor(), download=False)
test_set = MNIST(root='mnist', train=False, transform=transforms.ToTensor(), download=False)

train_loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
test_set = DataLoader(test_set, batch_size=len(test_set), shuffle=False)

train_x, train_y = next(iter(train_loader))
test_x, test_y = next(iter(test_set))

train_x = train_x.numpy().reshape(train_x.shape[0], -1)
train_y = train_y.numpy()
train_y = np.eye(10)[train_y] 
test_x = test_x.numpy().reshape(test_x.shape[0], -1)
test_y = test_y.numpy()

softmax = Softmax(train_x, train_y)
loss_list, acc_list = softmax.train(10, 0.1, batch_size=256)
# plt.plot(loss_list, color='blue')
# plt.plot(acc_list, color='red')
# plt.legend(['loss', 'accuracy'])
# plt

show_lineChart(loss_list, xlabel='epoch', ylabel='loss')
show_lineChart(acc_list, xlabel='epoch', ylabel='accuracy')

y_hat = np.argmax(softmax(test_x), axis=1)
print(np.equal(y_hat, test_y).sum() / test_y.shape[0])

# visualize
rand_idx = np.random.randint(0, 9989)
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.title('{} / {}'.format(y_hat[i+rand_idx], test_y[i+rand_idx]))
    plt.imshow(test_x[i+rand_idx, :].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.show()