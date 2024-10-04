# Learning Parity with Chain-of-Thought

This repository contains the code to reproduce the results from our paper *From Sparse Dependence to Sparse Attention: Unveiling How Chain-of-Thought Enhances Transformer Sample Efficiency*.

## Prerequisites

* [PyTorch](https://pytorch.org/get-started/locally/)
* [transformers](https://github.com/huggingface/transformers) 

## Data Generation

To generate synthetic parity data, first create a configuration file `data_config.py`. Below is an example configuration (`configs/data_config.py`): 

```python{
{
    'dataset_type': 'BinaryDataset',
    'n_samples': 10000,
    'val_samples': 2048,
    'n_digit': 30,
    'n_secret': 3,
    'use_cot': True,
}
```

- `n_samples`: Number of training samples.
- `val_samples`: Number of validation samples.
- `n_digit`: Number of input variables.
- `n_secret`: Number of secret variables.
- `use_cot`: Set to `True` to use Chain-of-Thought (CoT) data.

After defining your configuration, generate the data by running:

```bash
python src/generate_data.py --config path-to-data_config.py
```



## Training

To train a transformer model on the parity problem, use the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
PROJECT=Istree
MODEL=transformer
total_training_sample=100000
training_samples=10000
n_digits=30
k=3
CoT=True
lr=6e-5
num_layers=4
num_heads=3
python src/train.py \
    --world_size 1 \
    --total_training_samples ${training_samples} \
    --model_type transformer \
    --model_config_path config/gpt2_tiny_wpetrain.py \
    --dataset_dir data/Nonintersect_Binary/binary_${training_samples}_${n_digits}_${k}_${CoT}_False_False \
    --dataset_type BinaryDataset \
    --output_dir model/ \
    --batch_size 512 \
    --lr ${lr} \
    --weight_decay 0 \
    --log_interval 2048 \
    --save_interval 2048 \
    --eval_interval 2048 \
    --report_to_wandb \
   --num_hidden_layers ${num_layers} \
   --num_attention_heads ${num_heads} \
```

Where:

- `training_samples`: Number of samples in the training set.
- `total_training_samples`: Total number of samples used during training (`iterations = total_training_samples / batch_size`).
- `lr`: Learning rate.
- `num_layers`: Number of hidden layers in the model.
- `num_heads`: Number of attention heads.
- `CoT`: Set to `True` to train with Chain-of-Thought data, or `False` otherwise.

Training results will be saved in the `model/` directory.

## Reproducing the Results

### Figure 1, 2

To reproduce the results shown in Figures 1 and 2 of our paper, follow these steps:

1. Use the following configurations to generate data:

| n_samples  | n_digits | n_secret  | use_cot     |
| ---------- | -------- | --------- | ----------- |
| $10000000$ | $30$     | $1,2,3,4$ | True, False |

2. Train the model with the following settings:

| total_training_samples | training_samples | n_digits | k         | CoT         | num_layers | num_heads | lr                                              |
| ---------------------- | ---------------- | -------- | --------- | ----------- | ---------- | --------- | ----------------------------------------------- |
| $10^7$                 | $10^7$           | $30$     | $1,2,3,4$ | True, False | $1,2,3,4$  | $1,2,3,4$ | $6\times10^{-5}, 8\times10^{-5},1\times10^{-4}$ |

3. To reproduce Figure 1, run`Figures/Fig1/fig1.1.py` and `Figures/Fig1/fig1.2.py`.

   To reproduce Figure 2, run `Figures/Fig2/fig2.1.py`, `Figures/Fig2/fig2.2.py` and `Figures/Fig2/fig2.3.py`.

### Figure 3, 8

1. Use the following configurations to generate data:

| n_samples  | n_digits | n_secret           | use_cot |
| ---------- | -------- | ------------------ | ------- |
| $1000000$  | $100$    | $20,30,\cdots,100$ | True    |
| $10000000$ | $40$     | $20$               | True    |

2. Train the model with the following settings:

| total_training_samples | training_samples | n_digits | k                  | CoT  | num_layers | num_heads | lr                                              |
| ---------------------- | ---------------- | -------- | ------------------ | ---- | ---------- | --------- | ----------------------------------------------- |
| $10^7$                 | $10^6$           | $100$    | $20,30,\cdots,100$ | True | $1$        | $1$       | $6\times10^{-5}, 8\times10^{-5},1\times10^{-4}$ |
| $10^7$                 | $10^7$           | $40$     | $20$               | True | $1$        | $1$       | $6\times10^{-5}$                                |
| $10^7$                 | $10^7$           | $40$     | $20$               | True | $2$        | $2$       | $6\times10^{-5}$                                |

3. To reproduce Figure 3, run `Figures/Fig2/fig3.1.py` and`Figures/Fig2/fig3.2.py`.

### Figure 4

1. Use the following configurations to generate data:

| n_samples | n_digits | n_secret | use_cot     |
| --------- | -------- | -------- | ----------- |
| $10000$   | $20$     | $6$      | True, False |

2. Train the model with the following settings:

| total_training_samples | training_samples | n_digits | k    | CoT   | num_layers | num_heads | lr                |
| ---------------------- | ---------------- | -------- | ---- | ----- | ---------- | --------- | ----------------- |
| $10^7$                 | $10000$          | $20$     | $6$  | False | $1,2,3,4$  | $1,2,3,4$ | $1\times10^{-4}$  |
| $10^7$                 | $10000$          | $20$     | $6$  | True  | $1$        | $1$       | $1\times 10^{-5}$ |

3. To reproduce Figure 4, run`Figures/Fig4/fig4.py`.

### Figure 5, 10, 11

1. Use the following configurations to generate data:

| n_samples                    | n_digits | n_secret | use_cot |
| ---------------------------- | -------- | -------- | ------- |
| $5000,10000,50000,10^5,10^6$ | $20$     | $6$      | False   |

2. Train the model with the following settings:

| total_training_samples | training_samples             | n_digits | k    | CoT   | num_layers | num_heads | lr               |
| ---------------------- | ---------------------------- | -------- | ---- | ----- | ---------- | --------- | ---------------- |
| $10^7$                 | $5000,10000,50000,10^5,10^6$ | $20$     | $6$  | False | $1$        | $2$       | $1\times10^{-4}$ |
| $10^7$                 | $5000,10000,50000,10^5,10^6$ | $20$     | $6$  | False | $2$        | $3$       | $1\times10^{-4}$ |
| $10^7$                 | $5000,10000,50000,10^5,10^6$ | $20$     | $6$  | False | $4$        | $4$       | $1\times10^{-4}$ |

3. To reproduce Figure 5, run `Figures/Fig5/fig5.1.py`, `Figures/Fig5/fig5.2.py`.

   To reproduce Figure 10, run `Figures/Fig10/fig10.py`

   To reproduce Figure 11, run `Figures/Fig11/fig11.py`

### Figure 9

1. Use the following configurations to generate data:

| n_samples        | n_digits | n_secret | use_cot |
| ---------------- | -------- | -------- | ------- |
| $10^4,10^5,10^6$ | $20$     | $12$     | False   |
| $10^4,10^6$      | $20$     | $12$     | True    |

2. Train the model with the following settings:

| total_training_samples | training_samples | n_digits | k    | CoT  | num_layers    | num_heads     | lr                                               |
| ---------------------- | ---------------- | -------- | ---- | ---- | ------------- | ------------- | ------------------------------------------------ |
| $10^7$                 | $10^4,10^5,10^6$ | $20$     | $12$ | True | $1,2,3,4,6,8$ | $1,2,3,4,6,8$ | $6\times10^{-5}, 8\times10^{-5},1\times10^{-4}$  |
| $10^7$                 | $10^4,10^6$      | $20$     | $12$ | True | $1$           | $1$           | $6\times10^{-5},8\times 10 ^{-5},1\times10^{-4}$ |

3. To reproduce Figure 9, run `Figures/Fig5/fig9.py`.

### Figure 6,7

To reproduce Figure 6 and 7, first run `Figures/Fig6-7/work.py` to compute the normalized attention entropy. Then run `Figures/Fig6-7/fig6.py` and `Figures/Fig6-7/fig7.py`.