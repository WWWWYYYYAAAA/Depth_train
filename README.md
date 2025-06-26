### 模型定义在[ResCNN.py](./ResCNN.py)
### 训练函数在[train.py](./train.py)
### 损失函数在[loss.py](./loss.py)

### 参考工作：
#### 模型
    depth anything V2
    Zoedepth
    Distill any depth
    depth anything V2 Zoedepth 的对比
#### 数据集
    Kitti & NYU depth
    出一些的图imge-depth
### 我们的工作：
#### 模型设计
	网络
	损失函数
#### 训练
##### 方法
	Weight-decay，lr的配置
    优化器AdamW
##### 数据集处理
	采用depth anything small作为教师模型生成RGB图和深度图对
    从视频、照片、数据集中获取RGB图
    数据集中本身的RGB-深度图，裁剪旋转提升泛化能力
##### 训练结果
	图组
	不足分析
