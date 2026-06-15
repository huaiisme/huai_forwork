import random 
import torch

def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y1 = torch.matmul(X, w) + b
    y2 = X @ w + b 
    y1 += torch.normal(0, 0.01, y1.shape)
    
    return X, y1

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print(labels.shape)

def my_func():
    print("func start")
    return 1
    print(" never ")

result = my_func()
print(result)

# 理解yield生成器
def my_generator():
    print("first")
    yield 1
    print("second")
    yield 2
    print("third")
    yield 3 

g = my_generator()
next(g) # print("first")
next(g) 
next(g) # 
# next(g) # 抛出异常

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
gg = data_iter(batch_size, features, labels)
for X, y in gg:
    # print(X, '\n', y)
    pass
    # break

w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
print(w)
print(b)

def linreg(X, w, b):
    return torch.matmul(X, w) + b 

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    "小批量随机梯度下降"
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y) # X' 和 y'的小批量损失
        # 因为l的形状是(batch_size, 1)，而不是一个标量'l'中的所有元素被加到
        l.sum().backward()
        sgd([w, b], lr, batch_size) # 使用参数的梯度更新
    
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')


print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')









