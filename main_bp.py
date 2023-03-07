import torch
import torch.nn as nn
from d2l import torch as d2l

from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

def MNIST_loaders(train_batch_size=100, test_batch_size=100):

    transform = Compose([
        ToTensor(),
        # Normalize((0.1307,), (0.3081,)),  # 暂时不需要 normalize
        Lambda(lambda x: torch.flatten(x))])

    train_loader = DataLoader(
        MNIST('./datasets/', train=True,
              download=True,
              transform=transform),
        batch_size=train_batch_size, shuffle=True)

    test_loader = DataLoader(
        MNIST('./datasets/', train=False,
              download=True,
              transform=transform),
        batch_size=test_batch_size, shuffle=False)

    return train_loader, test_loader


class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":

    torch.manual_seed(1234)
    num_epochs = 20
    lr = 1e-3

    train_loader, test_loader = MNIST_loaders()
    net = Net().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        net.train()
        print(f"EPOCH: {epoch}")

        for X, y in train_loader:
            optimizer.zero_grad()

            X, y = X.cuda(), y.cuda()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()

            optimizer.step()

    