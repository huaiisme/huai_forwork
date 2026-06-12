import torch
import random

def synthetic_data():
    return

num_examples =              # 1000
true_w =                    # [2, -3.4]
true_b =                    # 4.2
features, labels =          # synthetic_data    

batch_size =                # 10


def data_iter



lr =                        # 0.03
num_epochs =                # 3
net =                       # need to def func linreg
loss =                      # need to def func squared_loss

# need to write train phase and need write sgd
for epoch in range(num_epochs):
    pass



print(f'w的估计误差:{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差:{true_b - b}')
