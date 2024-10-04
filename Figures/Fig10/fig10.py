import matplotlib.pyplot as plt
import numpy as np
import json
import os

def get_result(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            path = item_path
    path = os.path.join(path,"trainer_state.json")
    with open(path, 'r') as f:
        result = json.load(f)
    return result

def get_data(result):
    accuracy_, training_loss_, evaluation_loss_, entropy_, num_epoch_ = [], [], [], [], []
    for index, result_step in enumerate(result["log_history"]):
        if index % 2 == 1:
            accuracy_.append(result_step["eval_exact_match"])
            evaluation_loss_.append(result_step["eval_loss"])
            entropy_.append(result_step["mean_min_attention_entropy_over_token"])
            num_epoch_.append(result_step["epoch"])
        else:
            training_loss_.append(result_step["loss"])
    return accuracy_, training_loss_, evaluation_loss_, entropy_, num_epoch_
accuracy_color = '#1f77b4'  
training_loss_color = '#ff7f0e'  
evaluation_loss_color = '#7f7f7f'  

n_digits = 20
lr = 1e-4
total_samples = 10000000
num_layers = 4
num_heads = 4
(embedding, k) = ("gpt2_tiny_wpetrain",6)
fig, axs = plt.subplots(5, 3, figsize=(12, 16))  
for j,(num_layers, num_heads) in enumerate([(1,2), (2,3), (4,4)]):
    all_lines = []
    all_labels = []
    for i in range(5):
        n_samples = [5000, 10000, 50000, 100000, 1000000][i]
        path = f"model/binary_{n_samples}_{n_digits}_{k}_False_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
        result = get_result(path)
        accuracy_, training_loss_, evaluation_loss_, entropy_, num_epoch_ = get_data(result)
        steps_ = [item * n_samples / 512 for item in num_epoch_]
        
        ax1 = axs[i,j]
        ax1.plot(steps_, accuracy_, color=accuracy_color, label='Evaluation Accuracy')
        ax1.set_xlabel('Steps', fontsize=14)
        if j == 0:
            ax1.set_ylabel(f'{n_samples} training samples\nAccuracy', fontsize=16)
        else:
            ax1.set_ylabel(f'Accuracy', fontsize=16)
        
        
        ax1.set_ylim(0.48, 1.05)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)
        
        ax2 = ax1.twinx()
        ax2.plot(steps_, training_loss_, color=training_loss_color, label='Training Loss')
        ax2.plot(steps_, evaluation_loss_, color=evaluation_loss_color, label='Evaluation Loss')
        ax2.set_ylabel('Loss', fontsize=16)
        ax2.set_ylim(0, 1.2)
        ax2.tick_params(axis='y', labelsize=12)
        
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        if i == 0:
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            all_lines.extend(lines_1 + lines_2)
            all_labels.extend(labels_1 + labels_2)
        
        if i == 0:
            ax1.set_title(f"{num_layers} layer{'s' if num_layers != 1 else ''} {num_heads} heads", fontsize=18)


fig.legend(all_lines, all_labels, loc='upper center', bbox_to_anchor=(0.5,0), fontsize=17, frameon=False)

plt.tight_layout()
plt.savefig(f'Figures/Figs/fig10.pdf', dpi=300, bbox_inches='tight')
plt.savefig(f'Figures/Figs/flg10.svg', dpi=300, bbox_inches='tight')
plt.show()
