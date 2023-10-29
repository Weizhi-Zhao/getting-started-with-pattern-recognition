from le_net import LeNet
from torch import optim
import torch
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# dataset
train_set = MNIST(root='mnist', train=True, transform=transforms.ToTensor(), download=True)
test_set = MNIST(root='mnist', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_set, batch_size=256, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=True)

# train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = LeNet().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
loss_func = torch.nn.CrossEntropyLoss()

loss_list = []
acc_list = []

for epoch in range(10):
    tot_loss = 0
    tot_acc = 0
    tot_len = 0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        y_hat = net(images)
        loss = loss_func(y_hat, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tot_loss += loss.item()
        tot_acc += (torch.argmax(y_hat, dim=1) == labels).sum().item()
        tot_len += len(labels)
    loss_list.append(tot_loss / len(train_loader))
    acc_list.append(tot_acc / tot_len)

plt.plot(loss_list, color='blue')
plt.show()
plt.plot(acc_list, color='red')
plt.show()

with torch.no_grad():
    tot_acc = 0
    tot_len = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        y_hat = net(images)
        tot_acc += (torch.argmax(y_hat, dim=1) == labels).sum().item()
        tot_len += len(labels)
    print(tot_acc / tot_len)

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.title('{} / {}'.format(np.argmax(y_hat[i].cpu().numpy()), labels[i].cpu().numpy()))
        plt.imshow(images[i, :].cpu().numpy().reshape(28,28), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
