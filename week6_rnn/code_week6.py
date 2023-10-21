import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def load_dataset(batch_size):

    """
    -功能: 
    使用torchvision.datasets下载所需数据集
    通过DataLoader类指定batch大小并返回tensor

    -任务:
    请完善此函数 对齐各方法所需的数据类型接口 并正确读取到数据
    """

    mnist_train = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True)
    print(f'Fashion mnist_train: {len(mnist_train)}\nFashion mnist_test: {len(mnist_test)}')
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class Net(nn.Module):

    """
    -功能:
    CNN框架

    -任务:
    请自行实现CNN的搭建
    """

    def __init__(self):
        pass
        # todo
        
    def forward(self, x):
        pass
        # todo
        return x

def train(model, train_loader, criterion, optimizer, epoch):

    """
    -功能:
    实现模型的训练过程

    -任务:
    请完善train函数
    """

    model.train()
    train_loss = 0
    for data, label in train_loader:
        pass
        # todo

    train_loss = train_loss/len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def val(model, test_loader, criterion, epoch):
    
    """
    -功能:
    实现模型训练完成后的验证过程

    -任务:
    请完善val函数
    """

    model.eval()
    val_loss = 0
    groundtruth_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            pass
            # todo

    val_loss = val_loss/len(test_loader.dataset)
    groundtruth_labels, pred_labels = np.concatenate(groundtruth_labels), np.concatenate(pred_labels)
    acc = np.sum(groundtruth_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))

def main():
    #===== 1. 参数配置 =====
    batch_size = 256
    lr = 1e-3
    epochs = 20

    
    #===== 2. 读入数据 =====
    train_loader, test_loader = load_dataset(batch_size)
    

    #===== 3. 初始化模型 =====
    model = Net()

    #===== 4. 初始化学习准则及优化器 =====
    # 指定训练loss为交叉熵loss  优化器默认为Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        train(model, train_loader, criterion, optimizer, epoch)
        val(model, test_loader, criterion, epoch)

if __name__ == '__main__':
    main()