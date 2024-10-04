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
    accuracy_, training_loss_, evaluation_loss_, num_epoch_ = [], [], [], []
    for index, result_step in enumerate(result["log_history"]):
        if index % 2 == 1:
            accuracy_.append(result_step["eval_exact_match"])
            evaluation_loss_.append(result_step["eval_loss"])
            num_epoch_.append(result_step["epoch"])
        else:
            training_loss_.append(result_step["loss"])
    return accuracy_, training_loss_, evaluation_loss_, num_epoch_

def my_plot(path, accuracy, num_epoch):
    colors = [plt.cm.Greens,  plt.cm.Blues, plt.cm.Purples, plt.cm.Reds, plt.cm.Oranges, plt.cm.Greys]
    n_groups = 4
    n_curves_per_group = 4
    font_size = 18  

    fig, (ax, ax_color) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [5, 1]})

    for i in range(n_groups):
        for j in range(n_curves_per_group):
            num_layers = [1, 2, 3, 4, 6, 8][i]
            num_heads = [1, 2, 3, 4, 6, 8][j]
            color = colors[i](j / n_curves_per_group * 0.8 + 0.3) 
            ax.plot(num_epoch[i][j], accuracy[i][j], color=color, label=f'{num_layers} layers, {num_heads} heads')
    ax.set_ylim(0.48, 1)
    ax.set_xlabel('epoch number', fontsize=font_size)
    ax.set_ylabel('Evaluation Accuracy', fontsize=font_size)

    ax.tick_params(axis='both', which='major', labelsize=font_size)
    # ax.grid()

    for i in range(n_groups):
        for j in range(n_curves_per_group):
            ax_color.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[i](j / n_curves_per_group * 0.8 + 0.3)))

    ax_color.set_aspect('equal') 
    ax_color.set_xlim(0, n_curves_per_group)
    ax_color.set_ylim(0, n_groups)
    ax_color.set_xticks(np.arange(n_curves_per_group) + 0.5)
    ax_color.set_yticks(np.arange(n_groups) + 0.5)
    ax_color.set_xticklabels([f'{[1, 2, 3, 4, 6, 8][j]}' for j in range(4)], fontsize=font_size)
    ax_color.set_yticklabels([f'{[1, 2, 3, 4, 6, 8][i]}' for i in range(4)], fontsize=font_size)
    ax_color.set_xlabel('Head number', fontsize=15)
    ax_color.set_ylabel('Layer number', fontsize=15)

    ax_color.set_position([0.72, 0.65, 0.2, 0.2])  
    ax.axvline(x=5, color='blue', linestyle='--', linewidth=2)

    xticks = list(ax.get_xticks())  
    if 5 not in xticks:
        xticks.append(5)
    breakpoint()
    xticks = xticks[2:] 
    ax.set_xticks(xticks)  
    for label in ax.get_xticklabels():
        if label.get_text() == '5':  
            label.set_color('blue')  
    ax.set_xticklabels([f'{int(tick)}' if tick != 20 else '5' for tick in xticks])
    ax.set_xlim(0,1000)


    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()


accuracy = [[[] for _ in range(6)] for _ in range(6)]
num_epoch = [[[] for _ in range(6)] for _ in range(6)]
n_digits = 20


accuracy = [[[] for _ in range(6)] for _ in range(6)]
num_epoch = [[[] for _ in range(6)] for _ in range(6)]

(embedding,k)  =  ("gpt2_tiny_wpetrain",6)
n_samples = 10000
total_samples = 10000000
for idx_layers in range(4):
    for idx_heads in range(4):
        lr = 1e-4 if idx_layers == 0 else 6e-5
        num_layers = [1,2,3,4,6,8][idx_layers]
        num_heads = [1,2,3,4,6,8][idx_heads]
        path = f"model/Experiment2.1/binary_{n_samples}_{n_digits}_{k}_False_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
        result = get_result(path)
        accuracy_, training_loss_, evaluation_loss_, num_epoch_ = get_data(result)
        if accuracy[idx_layers][idx_heads] == []:
            accuracy[idx_layers][idx_heads] = accuracy_
            num_epoch[idx_layers][idx_heads] = num_epoch_
        else:
            print(accuracy_[-1], accuracy[idx_layers][idx_heads])
            if accuracy_[-1] > accuracy[idx_layers][idx_heads][-1] or  num_epoch_[-1] < num_epoch[idx_layers][idx_heads][-1]: # 学得更好或学得更快
                accuracy[idx_layers][idx_heads] = accuracy_
                num_epoch[idx_layers][idx_heads] = num_epoch_
my_plot(f"Fugures/Figs/fig4.pdf",accuracy,num_epoch)