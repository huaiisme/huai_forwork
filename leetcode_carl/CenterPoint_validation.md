## 验证自有数据跑程序，
### s1. 构造模拟测试数据
生成一个car类别，并在35.2, 0, 0的位置上 虚拟构造一个3.9m， 1.6m宽, 1.5m高的车
```
np.random.seed(42)  # 固定随机种子，结果可复现
car_x_range = [35.2 - 3.9/2, 35.2 + 3.9/2]
car_y_range = [0 - 1.6/2, 0 + 1.6/2]
car_z_range = [0 - 1.5/2, 0 + 1.5/2]
points_x = np.random.uniform(car_x_range[0], car_x_range[1], 1000)
points_y = np.random.uniform(car_y_range[0], car_y_range[1], 1000)
points_z = np.random.uniform(car_z_range[0], car_z_range[1], 1000)
points_intensity = np.ones(1000) * 0.5
test_points = np.stack([points_x, points_y, points_z, points_intensity], axis=1)  # (1000,4)
```


### s2. 判断输入的合理性
首先模拟数据生成一个shape为（1000，4）大小的 其中前3维为x,y,z 第4维为intensity
然后根据体素voxel的x,y,z的range，去判断点云范围
```
x_mask = (points[:,0] >= VOXEL_X_RANGE[0]) & (points[:,0] < VOXEL_X_RANGE[1])
y_mask = (points[:,1] >= VOXEL_Y_RANGE[0]) & (points[:,1] < VOXEL_Y_RANGE[1])
z_mask = (points[:,2] >= VOXEL_Z_RANGE[0]) & (points[:,2] < VOXEL_Z_RANGE[1])
```
再判断其有效率
```
valid_points_ratio = np.sum(x_mask & y_mask & z_mask) / len(points)
```

### s2. 

