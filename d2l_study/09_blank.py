import torch
import torchvision

from torch.utils import data
from torchvision import transforms
# def use_svg_display():
#     """使用svg格式在Jupyter中显示绘图

#     Defined in :numref:`sec_calculus`"""
#     backend_inline.set_matplotlib_formats('svg')

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="./data", train=True,
                                                transform=trans,
                                                download=True)