import numpy as np
import torch 
from torch.utils import data


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    return X, y.reshape((-1,1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个Pytorch数据迭代器
    """
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

# 使用框架的预定义好的层

from torch import nn 
net = nn.Sequential(nn.Linear(2, 1)) # 输入是2 输出是1 是一个list of layers

net[0].weight.data.normal_(0, 0.01) # 这里是用normal来替换掉data的值
net[0].bias.data.fill_(0) # p偏差设置成0 这两个模块等价实现设置w和b

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')



