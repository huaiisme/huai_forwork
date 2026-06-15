import torch
import numpy as np 
from torch.utils import data

def synthetic(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    # return X, y
    return X, y.reshape((-1,1))                 #   如果不加这一步结果不对

num_examples = 1000
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic(true_w, true_b, num_examples)

# 迭代器
def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个pytorch迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

from torch import nn 
net = nn.Sequential(nn.Linear(2, 1)) # 输入是2 输出是1 是一个list of layers

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss{l:f}')










