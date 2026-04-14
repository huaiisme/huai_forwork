### 验证自有数据跑程序，


首先模拟数据生成一个shape为（1000，4）大小的 其中前3维为x,y,z 第4维为intensity
然后根据体素voxel的x,y,z的range，去判断点云范围
```
x_mask = (points[:,0] >= VOXEL_X_RANGE[0]) & (points[:,0] < VOXEL_X_RANGE[1])
y_mask = (points[:,1] >= VOXEL_Y_RANGE[0]) & (points[:,1] < VOXEL_Y_RANGE[1])
z_mask = (points[:,2] >= VOXEL_Z_RANGE[0]) & (points[:,2] < VOXEL_Z_RANGE[1])
```
