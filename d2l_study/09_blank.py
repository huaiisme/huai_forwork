import torch
import torchvision

from torch.utils import data
from torchvision import transforms
from matplotlib import pyplot as plt
import time
import numpy as np

trans = transforms.ToTensor()

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签

    Defined in :numref:`sec_fashion_mnist`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# plt.show()

def get_dataloader_workers():
    """ 使用4个进程来读取的数据 """
    return 4

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()



def load_data_fashion_mnist



# 数据集、模型、工具函数可以放顶层
if __name__ == "__main__":
    batch_size = 256
    mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                    transform=trans,
                                                    download=True)

    mnist_test = torchvision.datasets.FashionMNIST(root="./data", train=False,
                                                    transform=trans,
                                                    download=True)

    print(len(mnist_train))
    print(len(mnist_test))

    print(mnist_train[0][0].shape)          # torch.Size([1, 28, 28]) 因为是黑白图片所以channel是1

    # 多进程DataLoader只在主进程执行
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                num_workers=get_dataloader_workers())


    # X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    # show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

    # 训练循环也要放进来
    timer = Timer()
    for X, y in train_iter:
        continue

    print(f'{timer.stop():2f} sec')
