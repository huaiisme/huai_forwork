import torch
import sys
from pathlib import Path
from torch import nn

# 获取当前脚本所在文件夹的上一级（项目根目录）
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# from my_d2l import get_dataloader_workers, load_data_fashion_mnist
from my_d2l import *


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


if __name__ == "__main__":

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
    net.apply(init_weights)
    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
