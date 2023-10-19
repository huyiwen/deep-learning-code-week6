import os
import torch
import argparse
import numpy as np

from models import TModel
from trainers import TModelTrainer
from utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloder, get_rating_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)

    # model args
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
    parser.add_argument("--intermediate_size", default=256, type=int, help="intermediate size of model")
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)
    parser.add_argument("--no_cuda", default=False, type=bool)

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print(torch.cuda.device_count())

    # --- 1.获取用户序列字典 & 最大序列长度---
    seq_dic, max_item = get_seq_dic(args)
    args.item_size = max_item + 1

    # 配置日志文件,记录训练过程
    cur_time = get_local_time()
    args_str = f'{args.data_name}-{cur_time}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # --- 2.读取数据 载入dataloader---
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    # --- 3.初始化模型---
    model = TModel(args=args)
    trainer = TModelTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)

    # --- 4.开始训练 ---
    for epoch in range(args.epochs):
        trainer.train(epoch)
        scores, _ = trainer.valid(epoch)

    print("---------------Sample 99 results---------------")
    # --- 5.测试 ---
    scores, result_info = trainer.test(0)

    print(result_info)
    # 记录测试结果
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

if __name__ == '__main__':
    main()
