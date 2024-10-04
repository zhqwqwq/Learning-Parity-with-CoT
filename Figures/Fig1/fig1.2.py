import matplotlib.pyplot as plt
import numpy as np
import json
import os

def get_result(path):
    for item in os.listdir(path):
        if item == "tmp":
            continue
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            path = item_path
            path = os.path.join(path,"trainer_state.json")
            if os.path.isfile(path):
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

n_digits = 30
lr = 1e-4
total_samples = 10000000
n_samples = 10000000

embedding = "gpt2_tiny_wpetrain"
k = 3
(num_hidden_layers, num_attention_heads) = (4,4)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
plt.subplots_adjust(wspace=0.7)

all_lines = []
all_labels = []
lr = 1e-4

for i in range(2):
    CoT = ["True", "False"][i]
    path = f"model/binary_{n_samples}_{n_digits}_{k}_{CoT}_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_hidden_layers}_#head={num_attention_heads}"
    result = get_result(path)
    accuracy_, training_loss_, evaluation_loss_, entropy_, num_epoch_ = get_data(result)
    entropy_ = np.array(entropy_)

    axs[i].set_title("with CoT" if CoT == "True" else "without CoT", fontsize=17)
    total_samples_ = [item * n_samples  for item in num_epoch_]

    idlayer = 0 
    lth = len(total_samples_) // 10 
    ax2 = axs[i].twinx()  

    line_accuracy, = ax2.plot(total_samples_[:lth], accuracy_[:lth], label='Evaluation Accuracy', 
                            color='red', linewidth=2, linestyle='--', alpha = 0.6)

    ax2.set_ylim(0.45, 1.1)
    ax2.set_ylabel('Evaluation Accuracy', fontsize=17)
    ax2.tick_params(axis='y', labelsize=14)  

    colors = plt.cm.viridis(np.linspace(0, 1, num_attention_heads)) 
    for j, color in zip(range(num_attention_heads), colors):
        line, = axs[i].plot(total_samples_[:lth], entropy_[:, idlayer, j][:lth], label=f'head {j+1}',
                            linewidth=2, color=color, alpha=0.8)  
        
        if i == 0 and idlayer == 0: 
            all_lines.append(line)
            all_labels.append(f'head {j+1}')

        axs[i].set_ylabel('Normalized Attention Entropy', fontsize=17)
        axs[i].set_xlabel('Iterations × Batch Size', fontsize=17)
        axs[i].set_ylim(0, 1.1)
        axs[i].set_xticks([0, 500000, 1000000])
        axs[i].tick_params(axis='both', labelsize=14) 
        axs[i].grid(True, which='both', linestyle='--', linewidth=0.5)

    all_lines.append(line_accuracy)
    all_labels.append('Accuracy')

    if i == 0:
        axs[i].legend(all_lines, all_labels, fontsize=12, loc='right')

plt.tight_layout()
plt.show()
plt.savefig(f'Figures/Figs/fig1.2.pdf')
plt.savefig(f'Figures/Figs/fig1.2.svg')