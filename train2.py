
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torchvision import transforms
import random
import os
# from mnist_dataset import MnistData
# from MS_depth import MSNet, MSloss, MSNet_fix, DepthLoss, MSNet_fix2, log_loss
from ResCNN import ResNetUNet
import time
import h5pickle
from torchvision.transforms import Compose
# from distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
from loss import DepthLossL3

import torchvision.transforms.functional as TF


class H5Dataset(Dataset):
    def __init__(self, file_path, image_key='images', label_key='depths', label_key2='labels'):
        self.file = h5pickle.File(file_path, 'r')
        self.images = self.file[image_key]
        self.labels = self.file[label_key]
        self.labels2 = self.file[label_key2]
       

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        label = torch.from_numpy(self.labels2[idx]).float()
        label = (label - label.min()) / (label.max() - label.min())
        return (
            torch.from_numpy(self.images[idx]).float()/255,
            torch.from_numpy(self.labels[idx]).float()/10,
            torch.from_numpy(self.labels2[idx]).float()/self.labels2[idx].max()
        )
        
class H5DatasetLarge(Dataset):
    def __init__(self, file_path, image_key='images', label_key='depths'):
        self.file = h5pickle.File(file_path, 'r')
        self.images = self.file[image_key]
        self.labels = self.file[label_key]
     

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
  
        return (
            torch.from_numpy(self.images[idx]).float()/255,
            torch.from_numpy(self.labels[idx]).float()/255,
        )



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

outx = 200
outy = 200

resize_transform = transforms.Resize(
    size=(outx, outy),         # 目标尺寸
    interpolation=Image.BILINEAR  # 插值方法
)
resize_transform2 = transforms.Resize(
    size=(640, 480),         # 目标尺寸
    interpolation=Image.BILINEAR  # 插值方法
)
resize_IN = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



class FixedAngleRotation:
    def __init__(self, angle):
        self.angle = angle  # 角度列表，如 [0, 90, 180]

    def __call__(self, img):
        return TF.rotate(img, self.angle, expand=False)  # expand=True 避免裁剪
    
    
# 训练
def train(model, loss_func, optimizer, checkpoints, epoch):
    print('Train......................')
    # 记录每个epoch的loss和acc
    best_acc = 0
    best_loss = 100000
    best_epoch = 0
    # 训练过程
    for epoch in range(0, epoch):
        # 设置计时器，计算每个epoch的用时
        start_time = time.time()
        model.train()  # 保证每一个batch都能进入model.train()的模式
        # 记录每个epoch的loss和acc
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        # train_data =  train_data.to(device)
        for i, (inputs, labels) in enumerate(tqdm(train_data)):
            # print(batch_size)
            # print(i, inputs, labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = resize_transform(inputs)
            labels = resize_transform(labels)
           
            # 预测输出
            outputs = model(inputs).squeeze(dim=1)
            loss = loss_func(outputs, labels)
            # print(outputs)
            # 因为梯度是累加的，需要清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 优化器
            optimizer.step()
            train_loss += loss.item()
            # train_acc += acc.item()
        # 验证集进行验证
        train_loss_epoch = train_loss / (i+1)

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(tqdm(val_data)):
              
               
                inputs = inputs.to(device)
                labels = labels.to(device)
                labels = resize_transform(labels)
                inputs = resize_transform(inputs)
                # 预测输出
                outputs = model(inputs).squeeze(dim=1)
                # 计算损失
                loss = loss_func(outputs, labels)
               
                val_loss += loss.item()
             
        # 计算每个epoch的训练损失和精度
        # train_loss_epoch = train_loss / train_data_size*batch_size
        # train_acc_epoch = train_acc / train_data_size*batch_size
        # 计算每个epoch的验证集损失和精度
        val_loss_epoch = val_loss / val_data_size
        # val_acc_epoch = val_acc / val_data_size
        end_time = time.time()
        print(
            'epoch:{} | time:{:.8f} | train_loss:{:.8f} | val_loss:{:.8f}'.format(
                epoch,
                end_time - start_time,
                train_loss_epoch,
                val_loss_epoch))

        # 记录验证集上准确率最高的模型
        best_model_path = checkpoints + "/" + 'best_model' + '.pth'
        if val_loss_epoch <= best_loss:
            best_loss = val_loss_epoch
            best_epoch = epoch
            torch.save(model, best_model_path)
        torch.save(model, checkpoints + '/last.pth')
        # torch.save(model.state_dict(), checkpoints + '/last_val.pth')
        print('Best loss for Validation :{:.8f} at epoch {:d}'.format(best_loss, best_epoch))
        # 每迭代50次保存一次模型
        # if epoch % 50 == 0:
        #     model_name = '/epoch_' + str(epoch) + '.pt'
        #     torch.save(model, checkpoints + model_name)
    # 保存最后的模型
    torch.save(model, checkpoints + '/last.pt')


if __name__ == '__main__':
    # batchsize
    # bs = 5000
    # learning rate
    # lr = 0.00000001
    lr = 5e-7
    # epoch
    epoch = 5
    # checkpoints,模型保存路径
    checkpoints = 'CNN2'
    os.makedirs(checkpoints, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    random_mode = 0
    val_mode = 0
    #load .mat
    # file_path = "data/nyu_depth_v2_labeled.mat"
    # dataset = H5Dataset(file_path)
    # file_path = "data/ID_20000_40000.mat"
    file_path = "../Distill-Any-Depth/data/ID_0_20000.mat"
    file_path1 = "../Distill-Any-Depth/data/ID_20000_40000.mat"
    file_path2 = "../Distill-Any-Depth/data/NYU.mat"
    file_path3 = "../Distill-Any-Depth/data/person.mat"
    file_path4 = "../Distill-Any-Depth/data/ID_40000_60000.mat"
    file_path5 = "../Distill-Any-Depth/data/dance.mat"
    # file_path = "data/ID_0_20000.mat"
    dataset = H5DatasetLarge(file_path)
    dataset1 = H5DatasetLarge(file_path1)
    dataset2 = H5DatasetLarge(file_path2)
    dataset3 = H5DatasetLarge(file_path3)
    dataset4 = H5DatasetLarge(file_path4)
    dataset5 = H5DatasetLarge(file_path5)
    data_size = dataset.__len__()
    # val_data_size = int(data_size*0.1)
    val_data_size = 1000
    batch_size = 16
    # train_dataset = Subset(dataset, range(val_data_size, data_size))
    train_dataset = ConcatDataset([dataset3, dataset2, dataset5, dataset, dataset1, dataset4])
    train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    
    # val_dataset = Subset(dataset, range(val_data_size))
    val_dataset = dataset2
    val_data_size = val_dataset.__len__()
    val_data = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=16)
    train_data_size = train_dataset.__len__()
    # 加载模型
    # model_path = checkpoints+"/best_model.pth"
    # model_path = checkpoints+"/last22.pth"
    # model_path = "test/MS_80e.pth"
    # model_path = "MSNet/random3.pth"
    # model_path = "MSNet3/last.pth"
    # print("Model Path", model_path)
    # model = torch.load(model_path, weights_only=False)
    model = ResNetUNet(in_channels=3, out_channels=1)
    # model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(checkpoints+'/last.pth', weights_only=False).state_dict())
    
    # GPU是否可用，如果可用，则使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # outx = 160
    # outy = 120
    print(device)
    model.to(device)
    # 损失函数
    # loss_func = MSloss
    # loss_func = nn.CrossEntropyLoss()
    # loss_func = DepthLoss
    loss_func = DepthLossL3(device=device)
    # loss_func = nn.L1Loss()
    # loss_func = nn.SmoothL1Loss()
    # 优化器，使用SGD,可换Adam
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # # 训练
    train(model, loss_func, optimizer, checkpoints, epoch)
    # for _ in range(20):
    #     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # # 训练
    #     train(model, loss_func, optimizer, checkpoints, epoch)
    #     lr = lr * 0.5