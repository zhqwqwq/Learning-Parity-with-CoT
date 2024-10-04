import argparse
import torch
import numpy as np
import random
from data import dataset_type_list

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'rnn', 'hybrid', 'peft', 'old_peft', 'dreamer'])
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--dataset_type', type=str, default='BinaryDataset',choices = dataset_type_list)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_training_samples', type=int, default=200000)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=100000)
    parser.add_argument('--save_interval', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_samples', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--report_to_wandb', action='store_true')
    parser.add_argument('--model_config_path', type=str, default=None)
    parser.add_argument('--num_virtual', type = int, default = 10)
    parser.add_argument('--world_size', type = int, default = 4)    
    parser.add_argument('--save_dir', type=str, default = None)
    parser.add_argument('--eval_interval', type = int, default = 2000)
    parser.add_argument('--use_cot', type = bool, default = False)
    parser.add_argument('--gate_type', type = str, default = 'AOX')
    parser.add_argument('--position_sensitive', type = bool, default = False)
    parser.add_argument('--additional_bit', type = bool, default = False)
    parser.add_argument('--num_hidden_layers', type = int, default = 1)
    parser.add_argument('--num_attention_heads', type = int, default = 1)
    parser.add_argument('--message',type = str, default = None)
    parser.add_argument('--random_seed',type = int, default = 42)
    return parser.parse_args()