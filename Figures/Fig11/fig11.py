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

n_digits = 20
lr = 1e-4
total_samples = 10000000
num_layers = 2
num_heads = 3

(embedding, k) = ("gpt2_tiny_wpetrain", 6) 
fig, axs = plt.subplots(2, num_layers, figsize=(11, 10))
column_titles = []
for i in range(num_layers):
    column_titles.append(f"Layer {i+1}")
for col in range(num_layers):
    if num_layers == 1:
        axs[0].set_title(column_titles[col], fontsize=24, pad=20)
    else:
        axs[0, col].set_title(column_titles[col], fontsize=24, pad=20)
    

all_lines = []
all_labels = []

for i in range(2):
    n_samples = [50000, 1000000][i]
    path = f"model/binary_{n_samples}_{n_digits}_{k}_False_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
    result = get_result(path)
    accuracy_, training_loss_, evaluation_loss_, entropy_, num_epoch_ = get_data(result)
    entropy_ = np.array(entropy_)
    if num_layers == 1:
        axs[i].set_ylabel(f'{n_samples} training samples\n\nNormalized Attention Entropy', fontsize=22, labelpad=20)
    else:
        axs[i, 0].set_ylabel(f'{n_samples} training samples\n\nNormalized Attention Entropy', fontsize=22, labelpad=20)
    
    total_samples_ = [item * n_samples for item in num_epoch_]

    for idlayer in range(num_layers):
        if num_layers == 1:
            ax = axs[i]
        else:
            ax = axs[i,idlayer]
        for j in range(num_heads):
            line, = ax.plot(total_samples_, entropy_[:, idlayer, j], label=f'head {j}', linewidth=3)

            if i == 0 and idlayer == 0:
                all_lines.append(line)
                all_labels.append(f'head {j}')
        ax.set_xlabel('Iterations × Batch Size', fontsize=18)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', labelsize=18) 
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.xaxis.get_offset_text().set_fontsize(16)
fig.legend(all_lines, all_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=18, frameon=False)

plt.tight_layout()

plt.savefig(f'Figures/Figs/fig11.svg')
plt.savefig(f'Figures/Figs/fig11.pdf')
plt.show()
