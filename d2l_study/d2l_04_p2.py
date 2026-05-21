import os

# 在 当前文件夹 下创建 data 文件夹
os.makedirs('data', exist_ok=True)

# 在 当前文件夹/data/ 下创建 csv
data_file = os.path.join('data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

#一般读取csv 用的是pandas这个库 

import pandas as pd
data = pd.read_csv(data_file)
print(data)

#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000
#存在数据缺失， 典型的方法有插值和删除，这里考虑插值
# 1. 划分输入特征与标签
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2] #  integer location 用列的数字编号，把特征和标签分开
# 2. 用列均值填充数值型特征的缺失值
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean()) # 用列的均值将NaN数值给替换填充

print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
# 处理前
#    NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN
# 处理后
#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0           1          0
# 1       2.0           0          1
# 2       4.0           0          1
# 3       3.0           0          1
# 现在inputs和outputs中的所有条目都是数值类型，可以转换为张量格式

import torch 
# X, y = torch.tensor(inputs.values), torch.tensor(outputs.values) 报错

print(inputs)

inputs = inputs.astype(float)

# 6. 转torch张量
X = torch.tensor(inputs.values, dtype=torch.float32)
y = torch.tensor(outputs.values, dtype=torch.float32)

print("✅ 成功！")
print(X)
print(y)



