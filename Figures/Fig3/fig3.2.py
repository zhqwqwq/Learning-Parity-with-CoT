from transformers import AutoModelForCausalLM, AutoModel
import matplotlib.pyplot as plt
from seaborn import heatmap
import seaborn as sns
import numpy as np
import os
import sys
import json
import torch
import transformers
from matplotlib.colors import LinearSegmentedColormap
sys.path.append("src")
from data import load_dataset
from model import get_model

def load_dataset_(dataset_dir):
    dataset_type =  'BinaryDataset'
    val_dataset = load_dataset(os.path.join(dataset_dir, 'val'), dataset_type)
    return val_dataset

def load_model_(embedding, model_dir, num_layers, num_heads):
    model_config_path = f"config/{embedding}.py"
    model_args = eval(open(model_config_path).read())
    model_args["num_hidden_layers"] = num_layers
    model_args["num_attention_heads"] = num_heads
    print(model_args)
    model = get_model(
        **model_args
    )
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            model_dir = item_path
    model_dir = os.path.join(model_dir,'pytorch_model.bin')
    model.load_state_dict(torch.load(model_dir))
    return model

n_digits = 40
total_samples = 10000000
num_layers = 1
num_heads = 1
n_samples = 10000000
embedding = "gpt2_tiny_wpetrain"
CoT = "True"
lr = 6e-5
for (n_samples, num_layers, num_heads) in [(1000000,2,2),(1000000,4,4)]:
    for k in [20]:
        secret = [1,2,3,7,12,13,14,16,18,19,22,24,25,26,27,29,33,34,36,38]
        model_dir = f"model/binary_{n_samples}_{n_digits}_{k}_{CoT}_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
        dataset_dir = f"data/Nonintersect_Binary/binary_{n_samples}_{n_digits}_{k}_{CoT}_False_False"
        val_dataset = load_dataset_(dataset_dir)
        model = load_model_(embedding, model_dir, num_layers, num_heads)
        model.eval()

        num = 1
        with torch.no_grad():
            attention_sum = None
            for i in range(num):
                data = val_dataset[i]
                print(i)
                output = model(input_ids = torch.tensor(data['input_ids']).unsqueeze(dim = 0),output_attentions=True)
                attention = output.attentions
                attention = torch.stack(attention, dim=0).transpose(0, 1)
                if attention_sum is None:
                    attention_sum = attention[0]
                else:
                    attention_sum += attention[0]
            average_attention = attention_sum / num
        average_attention = average_attention.to(torch.float32)

        colors = ["#2E004E", '#E60073', '#FF9933', '#FFE4CC']
        cmap = LinearSegmentedColormap.from_list("purple_yellow", colors)
        if num_layers == 1 and num_heads == 1:
            fig, axs = plt.subplots(num_layers, num_heads, figsize=(22, 7.5))
            mask = np.zeros_like(average_attention[0][0][n_digits:], dtype=bool)
            for i in range(k):
                for j in range(k-i):
                    mask[i, -j-1] = True  
            heatmap_plot = heatmap(
                average_attention[0][0][n_digits:], 
                cmap=cmap, 
                ax=axs, 
                cbar=True, 
                square=True, 
                vmin=0, 
                vmax=1, 
                mask=mask,
                cbar_kws={'shrink': 1, 'ticks': np.linspace(0, 1, 6), 'pad' : 0.04}  
            )
            
            heatmap_plot.figure.axes[-1].tick_params(labelsize=20)

            xticks = axs.get_xticklabels()
            for j, label in enumerate(xticks):
                if j in secret: 
                    label.set_color("red")
                else:
                    label.set_color("black")

            yticks = axs.get_yticks()
            new_ytick_labels = [str(int(tick) + 30) for tick in yticks]
            axs.set_yticklabels(new_ytick_labels)
            axs.set_xticklabels(axs.get_xticklabels(), rotation=45, ha='right',rotation_mode='anchor')
            axs.set_yticklabels(axs.get_yticklabels(), rotation=45, ha='right')

            axs.tick_params(axis='both', which='major', labelsize=20)

            fig.tight_layout()      
            fig.subplots_adjust(right=1.1)
            fig.suptitle("Attention Pattern", fontsize=28, x=0.48, y=1) 
            plt.savefig(f'Figures/Figs/fig3.2.pdf') 
            plt.savefig(f'Figures/Figs/fig3.2.svg') 
            