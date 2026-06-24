import torch
import sys
from pathlib import Path
from torch import nn
# 获取当前脚本所在文件夹的上一级（项目根目录）
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

# from my_d2l import get_dataloader_workers, load_data_fashion_mnist
from my_d2l import *

# 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 +b1)
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss()

if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    num_inputs, num_outputs, num_hiddens = 784, 10, 256 # 选了784和10之间的数 256 num_hidden
    W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True))
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))
    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

    params = [W1, b1, W2, b2]

    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    

