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
for (embedding,k) in [("gpt2_tiny_wpetrain",6)]:
    
    all_lines = []
    all_labels = []
    for i in range(2):
        fig, axs = plt.subplots(1, 1, figsize=(4.8, 3.8)) 
        n_samples = [50000, 1000000][i]
        path = f"model/binary_{n_samples}_{n_digits}_{k}_False_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
        result = get_result(path)
        accuracy_, training_loss_, evaluation_loss_, entropy_, num_epoch_ = get_data(result)
        total_samples_ = [item * n_samples for item in num_epoch_]
        
        ax1 = axs
        line1, = ax1.plot(total_samples_, accuracy_, color=accuracy_color, label='Evaluation Accuracy')
        ax1.set_xlabel('Iterations × Batch Size', fontsize=15)
        ax1.set_ylabel(f'{n_samples} training samples\nEvaluation Accuracy', fontsize=17)
        ax1.set_ylim(0.48, 1.05)
        ax1.tick_params(axis='y', labelsize=12)
        ax1.tick_params(axis='x', labelsize=12)

        ax2 = ax1.twinx()
        line2, = ax2.plot(total_samples_, training_loss_, color=training_loss_color, label='Training Loss')
        line3, = ax2.plot(total_samples_, evaluation_loss_, color=evaluation_loss_color, label='Evaluation Loss')
        ax2.set_ylabel(f'Loss', fontsize=17)
        if i == 0:
            ax2.set_ylim(0, 4.2)
        else:
            ax2.set_ylim(0, 1.2)
        ax2.tick_params(axis='y', labelsize=12)

        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

        lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]

        if i == 1:
            ax1.legend(lines, labels, loc='right',  fontsize=10)

        plt.tight_layout()
        plt.savefig(f'Figures/Figs/fig6_{n_samples}.svg', dpi=300, bbox_inches='tight')
        plt.savefig(f'Figures/Figs/fig6_{n_samples}.pdf', dpi=300, bbox_inches='tight')
        plt.show()
