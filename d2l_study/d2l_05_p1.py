"""
线性代数实现
"""


# 线性代数

# 标量由只有一个元素的张量表示

import torch
# 定义两个标量张量（一维、仅1个元素）
x = torch.tensor([3.0])
y = torch.tensor([2.0])

# 四则运算 + 幂运算
# x + y, x * y, x / y, x**y
print(x + y)
print(x * y)
print(x / y)
print(x**y)

A = torch.arange(20).reshape(5, 4)

A.T

X = torch.arange(24).reshape(2, 3, 4)

import torch
# 1. 生成 0~19 的一维张量，重塑为 5行4列 的浮点型矩阵
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
# 2. clone()：深拷贝，分配**全新内存**，B与A完全独立
B = A.clone()
# 3. 输出原矩阵A、A+B的运算结果
A, A + B
print(A.numel())
A.mean(), A.sum() / A.numel() 
print(A)
A.mean(axis=0), A.sum(axis=0) / A.shape[0] # 沿着行压缩
# 输出：(tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))

sum_A = A.sum(axis=1, keepdims=True) #按行求和（每行内部所有列相加）
#若不加 keepdims=True，结果会变成一维张量 [6,22,38,54,70] , 就不能广播

A.cumsum(axis = 0) # 累加求和 cumulative sum（累积和）
# A = [
#   [0, 1, 2, 3],    # 第 0 行
#   [4, 5, 6, 7],    # 第 1 行
#   [8,9,10,11],     # 第 2 行
#   [12,13,14,15],   # 第 3 行
#   [16,17,18,19]    # 第 4 行
# ]

# 第0行 = [0, 1, 2, 3]             （自己）
# 第1行 = [0+4, 1+5, 2+6, 3+7]     = [4,6,8,10]
# 第2行 = [4+8, 6+9, 8+10,10+11]   = [12,15,18,21]
# 第3行 = [12+12,15+13,18+14,21+15] = [24,28,32,36]
# 第4行 = [24+16,28+17,32+18,36+19] = [40,45,48,51]

y = torch.ones(4, dtype=torch.float32)

x, y, torch.dot(x,y)    # 点积
torch.sum(x * y)   #两者等价

# Ax 是一个长度为m的列向量，其ith元素是点积 AiTx
A.shape, x.shape, torch.mv(A, x)

B = torch.ones(4, 3)
torch.mm(A, B) # 矩阵乘法

u = torch.tensor([3.0, -4.0])
torch.norm(u) # u是一个向量的话

# L1fanshu
torch.abs(u).sum() # u是一个向量的话的L1范数

# 矩阵范数
# Frobenius norm 矩阵元素的平方和的平方根
torch.norm(torch.ones(4, 9))



