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
    全连接神经网络框架

    -任务:
    请自行实现全连接神经网络框架的搭建
    """

    def __init__(
        self,
        image_size,
        channels,
        dim,
        num_labels,
        expansion_factor=4,
        dropout = 0.,
        dense = nn.Linear
    ):
        super().__init__()
        inner_dim = int(dim * expansion_factor)
        self.num_labels = num_labels

        self.hidden_layers = nn.Sequential(
            dense(image_size*image_size*channels, inner_dim),  # hidden layer 1
            nn.GELU(),
            nn.Dropout(dropout),
            dense(inner_dim, dim),  # hidden layer 2
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(dim, num_labels)  # output layer

    def forward(self, pixel_values, labels=None, loss_fct=None):
        batch_size, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size, -1)

        layer_output = self.hidden_layers(pixel_values)
        logits = self.classifier(layer_output)

        loss = None
        if labels is not None:
            if loss_fct is None:
                loss_fct = nn.CrossEntropyLoss()
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return (loss, logits) if loss is not None else (logits,)


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
    config = dict(
        batch_size=256,
        lr=1e-3,
        epochs=20,
        channels=1,
        num_labels=10,
        image_size=28,
        dim=512,
        dropout=0.1,
        expansion_factor=4,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    print(config)

    #===== 2. 读入数据 =====
    train_loader, test_loader = load_dataset(config['batch_size'])
    # print(next(iter(train_loader)))

    #===== 3. 初始化模型 =====
    model = Net(
        image_size=config['image_size'],
        channels=config['channels'],
        dim=config['dim'],
        num_labels=config['num_labels'],
        dropout=config['dropout'],
        expansion_factor=config['expansion_factor'],
    ).to(config['device'])
    wandb.init(project="week4", config=config)
    # summary(model, input_size=(batch_size, channels, image_size, image_size))
    print(model)

    #===== 4. 初始化学习准则及优化器 =====
    # 指定训练loss为交叉熵loss  优化器默认为Adam
    # PyTorch会自动把整数型的label转为one-hot型 以便计算交叉熵
    # 思考：使用交叉熵损失函数时，预测模型的输出层是否需添加softmax层? 为什么？  （这个不用提交， 答案是否）
    T_max = config['epochs'] * len(train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer, config['lr'])

    for epoch in range(1, config['epochs']+1):
        info = {}
        info.update(train(model, train_loader, criterion, optimizer, scheduler, epoch, config['device']))
        info.update(val(model, test_loader, criterion, epoch, config['device']))
        wandb.log(info)

if __name__ == '__main__':
    main()
