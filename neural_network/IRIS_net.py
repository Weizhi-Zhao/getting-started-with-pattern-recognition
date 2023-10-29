from torch import nn

class Net(nn.Module):
    def __init__(self, depth=2, width=10, activation=nn.ReLU()):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(4, width))
        for _ in range(depth-2):
            self.layers.append(nn.Linear(width, width))
        self.layers.append(nn.Linear(width, 3))
        self.sofrmax = nn.Softmax(dim=-1)
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        y_hat = self.sofrmax(x)
        return y_hat