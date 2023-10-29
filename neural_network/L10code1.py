import csv
import numpy as np
from IRIS_net import Net
from torch import optim
import torch
import matplotlib.pyplot as plt
import os

EPOCH = 50
BATCH_SIZE = 50
DEPTH = 4 # 至少是2
WIDTH = 30
ACTIVATION = torch.nn.ReLU()
OPTIMIZER = optim.Adam
LEARNING_RATE = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset
data = []
n_class = 0
with open('iris/iris.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader) # skip header
    class_name = None
    for row in reader:
        if row[5] != class_name:
            class_name = row[5]
            n_class += 1
        data.append(list(map(float, row[1:5])) + [n_class])

data = np.array(data)
data = data.reshape(3, 50, 5)
train_set = []
test_set = []
for i in range(data.shape[0]):
    np.random.shuffle(data[i])
    train_set.append(data[i][:30])
    test_set.append(data[i][30:])
train_set = torch.cat([torch.tensor(i) for i in train_set], dim=0).type(torch.float32)
test_set = torch.cat([torch.tensor(i) for i in test_set], dim=0).type(torch.float32)

id = 0
for DEPTH in range(2, 7):
    for WIDTH in range(5, 45, 5):
        # train
        net = Net(depth=DEPTH, width=WIDTH, activation=ACTIVATION).to(device)
        optimizer = OPTIMIZER(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        loss_func = torch.nn.CrossEntropyLoss()

        loss_list = []
        for epoch in range(EPOCH):
            n = train_set.size()[0]
            indices = torch.randperm(n)
            train_set = train_set[indices]
            for i in range(0, train_set.size()[0], BATCH_SIZE):
                batch_x = train_set[i:min(i + BATCH_SIZE, n), :-1].to(device)
                batch_y = train_set[i:min(i + BATCH_SIZE, n), -1].type(torch.int) - 1
                batch_y = torch.eye(n_class)[batch_y].to(device)
                y_hat = net(batch_x)
                loss = loss_func(y_hat, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())

        acc = 0
        # eval
        with torch.no_grad():
            y_hat = net(test_set[:, :-1].to(device))
            y_hat = torch.argmax(y_hat, dim=1).cpu()
            y = test_set[:, -1].type(torch.int) - 1
            acc = (y_hat == y).sum().item() / y.size()[0]
            print(acc)

        id += 1
        plt.subplot(5, 8, id)
        plt.title(f"{DEPTH}_{WIDTH}_{LEARNING_RATE}_{acc*100:.1f}")
        plt.plot(loss_list)
        plt.axis('off')
        # if os.path.exists('code1_res') is False:
        #     os.mkdir('code1_res')
        # plt.savefig(f"code1_res/{DEPTH}_{WIDTH}_{ACTIVATION.__class__.__name__}\
        #             _{OPTIMIZER.__name__}_{LEARNING_RATE}_{acc}.png".replace(' ', ''),
        #             dpi=300,
        #             )
        
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()
plt.tight_layout()
plt.show()
