# import numpy as np

# # 生成 1-16，reshape 成 4行4列
# matrix = np.arange(1, 17).reshape(4, 4)

import torch

x = torch.arange(16)
# print(x)

# print(x.shape)
matrix = x.reshape(4, 4)

# print(matrix)
# # [[ 1  2  3  4]
# #  [ 5  6  7  8]
# #  [ 9 10 11 12]
# #  [13 14 15 16]]
# print(matrix[1,2])
# print(matrix[1,:])
# print(matrix[:,1])
# print(matrix[1:3,1:])
# print(matrix[::3, ::2])


X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
CAT1 = torch.cat((X, Y), dim = 0)
CAT2 = torch.cat((X, Y), dim = 1)
print(CAT1)
print(CAT2)
print(X == Y)
X.sum()

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

print(a)
print(b)
print(a + b) # 广播机制
 
print(X[-1]) #最后一行
print(X[1:3]) #最后两行

X[1,2] = 9
print(X)
# 这样X[1,2]的值就变化了

X[0:2, :] = 12
print(X)
# 区域赋值

before = id(Y) # 类似C++指针
Y = Y + X
id(Y) == before

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])

print(a)
print(a.item())
print(float(a))
print(int(a))




