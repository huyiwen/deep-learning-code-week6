import torch
import torch.nn as nn
from modules import Encoder, LayerNorm

class TModel(nn.Module):
    def __init__(self, args):
        super(TModel, self).__init__()
        self.args = args
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # todo
        # 缺少LayerNorm、dropout
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position_embedding(self, sequence):
        """
        sequence.shape = [batch_size, len]
        """
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # todo
        # 1.sequence过embedding
        # 2.position_ids过embedding
        # 3.给item的embedding加上位置编码得到sequence_emb
        # 4.sequence_emb过LayerNorm、dropout
        item_embedding = self.item_embeddings(sequence)
        pos_embedding = self.position_embeddings(position_ids)
        sequence_emb = item_embedding + pos_embedding
        sequence_emb = self.dropout(self.LayerNorm(sequence_emb))
        return sequence_emb

    def forward(self, input_ids):

        sequence_emb = self.add_position_embedding(input_ids)
        # todo
        # 得到添加了位置编码后的sequence_emb后，即完成了transformer的前面两个模块，接下来送入我们构造的encoder(即改造后的注意力层)
        # 经过encoder后得到sequence_output
        sequence_output = self.item_encoder(sequence_emb)

        return sequence_output
