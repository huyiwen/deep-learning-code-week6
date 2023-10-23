# Week 6

我们采用类似 VGG 的架构, 将 `[B C H W]` 经过四次图片大小减半, Channel 数每层乘一定倍数. 经过 20 个 epoch 训练后, 不同的模型超参数可以达到 `93.14%` 和 `93.52%` 的准确率.

## 模型

首先构造每一层 VGG Block
```python
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

```

其次根据不同的参数构造相应的 VGG 模型, 最后使用分类器进行分类:
```python
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
```

## 训练, 评估和数据增广

训练和评估函数同第四周, 数据增广同第四周.
