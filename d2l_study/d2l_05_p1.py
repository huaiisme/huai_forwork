import torch
# 对 y=2xTx求导
x = torch.arange(4.0)
print(x)

# 计算梯度需要一个存梯度的地方
x.requires_grad_(True)
print(x.grad)
# 通过调用反向传播函数来自动计算y关于x每个分量的梯度
y = 2 * torch.dot(x, x) 
print(y)
y.backward()
print(x.grad) # tensor([ 0.,  4.,  8., 12.])


y = 2 * torch.dot(x, x) 
y.backward()
print(x.grad)   # tensor([ 0.,  8., 16., 24.])

# 默认情况下pytorch会累计梯度

x.grad.zero_() 
print(x.grad)
y = x * x
print(y)
print(y.sum())
y.sum().backward()

x.grad.zero_()
y = x * x
# u = y.detach()
u = y
z = u * x
z.sum().backward()
print(x.grad)
