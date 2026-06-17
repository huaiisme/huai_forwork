import torch
import torchvision
from torchvision import transforms
from torch.utils import data



def get_dataloader_workers():
    """ 使用4个进程来读取的数据 """
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """ 下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    # print(X_exp.sum(1, keepdim=True))
    return X_exp / partition # 这里应用了广播机制


def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) 
    # 需要的是一个批量大小乘以输入维度的矩阵，所以reshape成一个2d的矩阵 -1所在位置维度为批量大小，W.shape[0]为784
    # 所以返回的是一个256 x 784的矩阵

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]) #给定我y_hat的预测和真实标号y

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 评估在任意模型net的准确率
def evaluate_accuracy(net, data_iter):
    """ 计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 模型设置为评估模式
    metric = Accumulator(2) # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            updater.step()
            metric.add(
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel()
            )
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(
                float(l.sum()), accuracy(y_hat, y),
                y.size().numel()
            )
        return metric[0] / metric[2], metric[1] / metric[2]



def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
lr = 0.1

def updater(batch_size):
    return sgd([W, b], lr, batch_size)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

    train_loss, train_acc = train_metrics


if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    num_inputs = 784 # 28x28
    num_outputs = 10

    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # print(X.sum(0, keepdim=True))
    # print(X.sum(1, keepdim=True))
    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    # X_prob, X_prob.sum(1)

    # print(X)
    # print(X_prob)
    # print(X_prob.sum(1))

    # 创建一个数据y_hat, 其中包含2个样本在3个类别的预测概率，使用y作为y_hat中概率的索引
    y = torch.tensor([0, 2])    # 创建一个长度为2的向量，第一个为0 第二个是2 表示两个真实的标号
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # y_hat为预测值
    y_hat[[0, 1], y]  # 张量花式索引，会一一配对取元素 行索引和样本一一对应：第0行配y[0]，第1行配y[1]
    # tensor([0.1000, 0.5000])
    # print(cross_entropy(y_hat, y))
    # print(accuracy(y_hat, y) / len(y))

    # print(evaluate_accuracy(net, test_iter))
    num_epochs = 1
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)


