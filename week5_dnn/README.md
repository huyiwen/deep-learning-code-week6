# Week5

本次实验用 DNN 替代了 Attention 模块, 补全了一个类似 Transformers 模型, 在 `num_hidden_layers=2, num_attention_head=2` 架构下训练 1 个 epoch 后相关任务上达到了 `HIT@1=0.0709, NDCG@1=0.2206, HIT@5=0.2206, NDCG@5=0.1459, HIT@10=0.3214, NDCG@10=0.1784, MRR=0.1548` 的准确度, 在`num_hidden_layers=12, num_attention_head=6` 架构下训练 `epochs=10` 达到 `'HIT@1': '0.1749', 'NDCG@1': '0.1749', 'HIT@5': '0.3902', 'NDCG@5': '0.2868', 'HIT@10': '0.4964', 'NDCG@10': '0.3212', 'MRR': '0.2842'` 准确度.

## 模型

### 中间层

```python
self.linear = nn.Sequential(
	nn.Linear(args.hidden_size, args.intermediate_size),
	nn.ReLU(),
	nn.Linear(args.intermediate_size, args.hidden_size),
)
```


### Encoder

```python
self.layer = nn.Sequential(*[Layer(args) for _ in range(args.num_hidden_layers)])
```

### TModel

模型补全:
```python
self.item_encoder = Encoder(args)
self.LayerNorm = LayerNorm(args.hidden_size)
self.dropout = nn.Dropout(args.hidden_dropout_prob)
```

Position Embedding:
```python
item_embedding = self.item_embeddings(sequence)
pos_embedding = self.position_embeddings(position_ids)
sequence_emb = item_embedding + pos_embedding
sequence_emb = self.dropout(self.LayerNorm(sequence_emb))
```


