import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def load_dataset(batch_size):

    """
    -功能: 
    使用torchvision.datasets下载所需数据集
    通过DataLoader类指定batch大小并返回tensor

    -任务:
    请完善此函数 对齐各方法所需的数据类型接口 并正确读取到数据
    """

    horizontal = [transforms.RandomHorizontalFlip(1)]
    vertical = [transforms.RandomVerticalFlip(1)]
    augment = [transforms.RandAugment()]

    transform = [
        transforms.Grayscale(),
        transforms.ToTensor(),
    ]

    mnist_train1 = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transforms.Compose(horizontal+transform))
    mnist_train2 = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transforms.Compose(vertical+transform))
    mnist_train3 = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transforms.Compose(transform))
    mnist_train4 = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transforms.Compose(horizontal+vertical+transform))
    mnist_train5 = torchvision.datasets.FashionMNIST(root='./datasets', train=True, download=True, transform=transforms.Compose(augment+transform))
    mnist_train = mnist_train1 + mnist_train2 + mnist_train3 + mnist_train4 + mnist_train5

    mnist_test = torchvision.datasets.FashionMNIST(root='./datasets', train=False, download=True, transform=transforms.Compose(transform))

    print(f'Fashion mnist_train: {len(mnist_train)}\nFashion mnist_test: {len(mnist_test)}')

    train_loader = DataLoaderX(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=10, pin_memory=True)
    test_loader = DataLoaderX(mnist_test, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True)

    return train_loader, test_loader


class Net(nn.Module):

    """
    -功能:
    CNN框架

    -任务:
    请自行实现CNN的搭建
    """

    @staticmethod
    def vgg_block(num_hidden_layers, in_channels, out_channels):
        vgg_blocks = [
            nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ) if i % 2 == 0 else nn.ReLU() for i in range(num_hidden_layers * 2)
        ] + [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*vgg_blocks)

    def __init__(self, fc_features, fc_hidden_units=4096, dropout_prob=0.1, arch=[[1, 1, 8], [2, 8, 64], [2, 64, 512], [2, 512, 512]]):
        super().__init__()
        self.num_labels = 10
        self.vggs = nn.Sequential(*[self.vgg_block(*param) for param in arch])
        self.classifier = nn.Sequential(*[
            nn.Linear(fc_features, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(fc_hidden_units, self.num_labels),
        ])
        
    def forward(self, x, labels, criterion):
        # print(x.shape)  # B C L H
        hidden_states = self.vggs(x)
        # print(hidden_states.shape)  # B C L H
        flatten = hidden_states.flatten(start_dim=1)
        # print(flatten.shape)  # B X
        logits = self.classifier(flatten)

        loss = None
        if labels is not None:
            if criterion is None:
                criterion = nn.CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))
        return loss, logits

def train(model, train_loader, criterion, optimizer, scheduler, epoch, device):

    """
    -功能:
    实现模型的训练过程

    -任务:
    请完善train函数
    """

    model.train()
    train_loss = 0
    for data, label in tqdm(train_loader):
        data, label = data.to(device), label.to(device)
        loss, logits = model(data, label, criterion)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    return {"train_loss": train_loss, "lr": scheduler.get_last_lr()[0]}


def val(model, test_loader, criterion, epoch, device):
    
    """
    -功能:
    实现模型训练完成后的验证过程

    -任务:
    请完善val函数
    """

    model.eval()
    val_loss = 0
    groundtruth_labels = [ex[1].numpy() for ex in test_loader]
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            loss, logits = model(data, label, criterion)
            val_loss += loss.item()
            pred_labels.append(logits.argmax(dim=-1).detach().cpu().numpy())

    val_loss = val_loss / len(test_loader.dataset)
    groundtruth_labels, pred_labels = np.concatenate(groundtruth_labels), np.concatenate(pred_labels)
    acc = np.sum(groundtruth_labels == pred_labels) / len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))
    return {"val_loss": val_loss, "val_acc": acc}

def main():
    #===== 1. 参数配置 =====
    batch_size = 256
    lr = 1e-3
    epochs = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    #===== 2. 读入数据 =====
    train_loader, test_loader = load_dataset(batch_size)
    

    #===== 3. 初始化模型 =====
    model = Net(512, 4096).to(device)

    #===== 4. 初始化学习准则及优化器 =====
    # 指定训练loss为交叉熵loss  优化器默认为Adam
    T_max = epochs * len(train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    for epoch in range(1, epochs+1):
        train(model, train_loader, criterion, optimizer, scheduler, epoch, device)
        val(model, test_loader, criterion, epoch, device)

if __name__ == '__main__':
    main()