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

### s3. 体素化模块验证
```
test_pillars, test_coords = validate_voxelization(test_points)
```

对于每个points而言，首先对超出范围的点剔除掉
获得其mask，如果len(points)为0,则 返回一个 np.zeros((1, MAX_POINTS_PER_PILLAR, 9), dtype=np.float32), np.zeros((1, 3), dtype=np.int32)
按照以下方式，通过x对体素范围做相减再除以size的方式 确定x y z 的idx
```
# 2. 计算每个点所属的体素网格坐标
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]
x_idx = ((x - VOXEL_X_RANGE[0]) / VOXEL_SIZE[0]).astype(np.int32)
y_idx = ((y - VOXEL_Y_RANGE[0]) / VOXEL_SIZE[1]).astype(np.int32)
z_idx = ((z - VOXEL_Z_RANGE[0]) / VOXEL_SIZE[2]).astype(np.int32)
voxel_coords = np.stack([z_idx, y_idx, x_idx], axis=1)  # (N,3)
```
# 3. 去重，得到非空柱体
```
unique_coords, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
num_pillars = min(len(unique_coords), MAX_PILLARS)
unique_coords = unique_coords[:num_pillars]
```
获得pillar的数目
然后根据pillar的数目来对柱体的特征做初始化
# 4.初始化pillar的特征
```
pillars = np.zeros((num_pillars, MAX_POINTS_PER_PILLAR, 9), dtype=np.float32)
```
pillar的数目，以及每个pillar的最大点数，这里是32，以及9维信息

# 5. 填充每个柱体的点，计算增强特征
对于单个point
设置两个条件，当voxel idx大于num pillar的时候直接continue
当point_count[voxel_id] 大于pillar中的最大值的时候continue
读取point[i]
赋值voxel_x voxel_y voxel_z
```
# 柱体网格中心
  voxel_x = (unique_coords[voxel_idx, 2] * VOXEL_SIZE[0]) + VOXEL_X_RANGE[0] + VOXEL_SIZE[0]/2
  voxel_y = (unique_coords[voxel_idx, 1] * VOXEL_SIZE[1]) + VOXEL_Y_RANGE[0] + VOXEL_SIZE[1]/2
  voxel_z = (unique_coords[voxel_idx, 0] * VOXEL_SIZE[2]) + VOXEL_Z_RANGE[0] + VOXEL_SIZE[2]/2
```

# 填充9维特征
``` x_p y_p z_p i_p x_p - voxel_x, y_p - voxel_y, z_p - voxel_z, x_p - meanx, y_p - meany
pillar[voxel_idx, point_count[voxel_idx]] = [
            x_p, y_p, z_p, i_p,
            x_p - voxel_x, y_p - voxel_y, z_p - voxel_z,
            x_p - np.mean(x), y_p - np.mean(y)
        ]
```
这样得到了有效数目,32,9个矩阵维度

``` 判断偏移量
# 偏移量验证
x_offset = pillars[...,4]
y_offset = pillars[...,5]
assert (np.abs(x_offset) <= 0.051).all()
assert (np.abs(y_offset) <= 0.051).all()
```

### s4. 标签生成逐字段验证
将gt框给到用于生成heatmap与reg z dim rot
"""
生成CenterPoint训练所需的标签
输入：boxes_3d → (M,7) [x,y,z,l,w,h,ry]
输出：
    heatmap: (NUM_CLASSES, BEV_HEIGHT, BEV_WIDTH)
    reg: (2, BEV_HEIGHT, BEV_WIDTH)
    z: (1, BEV_HEIGHT, BEV_WIDTH)
    dim: (3, BEV_HEIGHT, BEV_WIDTH)
    rot: (2, BEV_HEIGHT, BEV_WIDTH)
"""
```
array([[35.2,  0. ,  0. ,  3.9,  1.6,  1.5,  0. ]], dtype=float32)
heatmap, reg, z, dim, rot = generate_targets(gt_boxes)
其中BEV HEIGHT与 BEV WIDTH是根据体素voxel x range 和 voxel y range所决定的
```
将3d框转为BEV参数
读取
```
"""3D框 → BEV平面的中心、尺寸、角度"""
bev_centers = boxes_3d[:, [0, 1]]  # x,y
bev_dims = boxes_3d[:, [3, 4]]     # l,w
ry = boxes_3d[:, 6]
```
再将 投影到BEV
先获得BEV框，再将BEV框的xy 即bev_center按照对应体素化的程序给到
然后通过对逐个 boxes_3d生成标签
首先是
```
u, v = pixel_coords[i]
# 过滤超出BEV范围的框
if u < 0 or u >= BEV_WIDTH or v < 0 or v >= BEV_HEIGHT:
    continue
```
根据bev l w生成高斯热力核
```
  # 1. 生成热力图高斯核
  l, w = bev_dims[i]
  # 米 → 像素
  l_pixel = l / 0.1
  w_pixel = w / 0.1
  radius = max(int(gaussian_radius((l_pixel, w_pixel))), 1)
  heatmap[0] = draw_gaussian(heatmap[0], (u, v), radius)
```
这里这个0.1应该是需要跟具体的业务去做一个匹配的

有了l和w，使用高斯核半径
```
def gaussian_radius(det_size, min_overlap=0.7):
    """计算高斯核半径（CenterNet/CenterPoint官方公式）"""
    l, w = det_size
    a1 = 1
    b1 = l + w
    c1 = w * l * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (l + w)
    c2 = (1 - min_overlap) * w * l
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (l + w)
    c3 = (min_overlap - 1) * w * l
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return max(r1, r2, r3)
```
通过给定的l和 W， 以及给出一个Min_overlap
分别划出3个r半径出来

然后有了 u v （BEVcenter上的x，y），以及刚计算得到的radius
然后使用gaussian = 得到
```
gaussian = multivariate_normal(mean=[0, 0], cov=[[diameter/6, 0], [0, diameter/6]])
```


















