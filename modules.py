import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """
    gelu
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        LayerNorm
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class FilterLayer(nn.Module):
    """
    滤波器层
    """
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        # todo
        # input_tensor.shape = [batch, seq_len, hidden]
        # 滤波器层可以参考文献[4] 的做法，仓库地址是:https://github.com/raoyongming/GFNet
        batch, seq_len, hidden = input_tensor.shape             # input_tensor.shape=[256, 50, 64]
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')   # 一维序列数据不用rfft2，快速傅里叶变换会让tensor的维度少一半+1，可以去看论文是怎么解释的,在section3.2
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight                                          # x.shape=[256, 26, 64];weight.shape=[1, 26, 64]一维序列数据滤波
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states    # hidden_states.shape=[256, 50, 64]

class Intermediate(nn.Module):
    """
    中间层
    """
    def __init__(self, args):
        super(Intermediate, self).__init__()
        # todo
        # 可以自己设计，就是一个前向网络
        # LayerNorm，Dropout
        self.LayerNorm = LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear = nn.Sequential(
            nn.Linear(args.hidden_size, args.intermediate_size),
            nn.ReLU(),
            nn.Linear(args.intermediate_size, args.hidden_size),
        )

    def forward(self, input_tensor):
        # todo
        # input_tensor.shape = [256, 50, 64]
        # hidden_states.shape = [256, 50, 64]
        # 注意框图里面的残差链接
        hidden_states = self.LayerNorm(input_tensor)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = hidden_states + input_tensor
        return hidden_states

class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)      # 中间层就是所给框图中的后面几层

    def forward(self, hidden_states):

        hidden_states = self.filterlayer(hidden_states)             # hidden_states.shape = [256, 50, 64]

        intermediate_output = self.intermediate(hidden_states)      # 拿到了隐层tensor过一个前向网络即可
        return intermediate_output

class Encoder(nn.Module):
    """
    根据自己搭建的Layer组成Transformer only-MLP 的Encoder
    """
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.layer = nn.Sequential(*[Layer(args) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, output_all_encoded_layers=True):
        # todo
        # 由于我们单独把encoder抽象成了一个类，因此这里可以叠多层encoder(和原始transformer的多头注意力一致)
        # 这里的all_encoder_layers是用来存每一层encoder的输出，即hidden_states，你可以选择只搭一层，直接返回，不要这个列表
        output = self.layer(hidden_states)

        return output
