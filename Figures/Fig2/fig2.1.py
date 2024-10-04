import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import json
import os

def get_result(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            path = item_path
    path = os.path.join(path,"trainer_state.json")
    if os.path.isfile(path):
        with open(path, 'r') as f:
            result = json.load(f)
        return result
    else:
        return None

def get_sample_complexity(result):
    for index, result_step in enumerate(result["log_history"]):
        if index % 2 == 1:
            if result_step["eval_exact_match"] == 1:
                # breakpoint()
                return result_step["epoch"] * 10000000
    return 10000000

n_digits = 30
n_samples = 10000000
total_samples = 10000000


def my_plot(sample_complexity):
    log_norm = colors.LogNorm(vmin=10**4, vmax=10**7)
    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 0.1])  

    axes = [fig.add_subplot(gs[0, i]) for i in range(4)] + [fig.add_subplot(gs[1, i]) for i in range(4)]

    for i in range(4):
        sns.heatmap(sample_complexity["True"][i], ax=axes[i], cmap="YlGnBu", norm=log_norm, cbar=False)
        axes[i].set_title(f'k = {i+1}', fontsize=24)
        axes[i].set_xlabel("Head Number", fontsize=24)
        axes[i].set_ylabel("Layer Number", fontsize=24)  
        axes[i].set_xticks([0.5, 1.5, 2.5, 3.5])
        axes[i].set_xticklabels([1, 2, 3, 4],fontsize=18)
        axes[i].set_yticks([0.5, 1.5, 2.5, 3.5])
        axes[i].set_yticklabels([1, 2, 3, 4],fontsize=18)
        if i == 0:
            axes[i].text(-1.3, 2, "with CoT", fontsize=26, rotation=90, va="center") 

    for i in range(4):
        sns.heatmap(sample_complexity["False"][i], ax=axes[i+4], cmap="YlGnBu", norm=log_norm, cbar=False)
        axes[i+4].set_xlabel("Head Number", fontsize=24)
        axes[i+4].set_ylabel("Layer Number", fontsize=24)  
        axes[i+4].set_xticks([0.5, 1.5, 2.5, 3.5])
        axes[i+4].set_xticklabels([1, 2, 3, 4],fontsize=18)
        axes[i+4].set_yticks([0.5, 1.5, 2.5, 3.5])
        axes[i+4].set_yticklabels([1, 2, 3, 4],fontsize=18)
        if i == 0:
            axes[i+4].text(-1.3, 2, "without CoT", fontsize=26, rotation=90, va="center")  

    cbar_ax = fig.add_subplot(gs[:, 4]) 
    cbar = fig.colorbar(axes[-1].collections[0], cax=cbar_ax)
    cbar.set_label('Sample Complexity', fontsize=26)
    cbar.ax.yaxis.set_tick_params(labelsize=22)

    def custom_formatter(x, pos):
        if x == 10**7:
            return '≥ 10⁷' 
        else:
            return f'$10^{{{int(np.log10(x))}}}$' 

    cbar.formatter = ticker.FuncFormatter(custom_formatter)
    cbar.update_ticks()  

    plt.tight_layout()
    plt.show()
    plt.savefig("Figures/Figs/fig2.1.pdf")
    plt.savefig("Figures/Figs/fig2.1.svg")


for embedding in ["gpt2_tiny_wpetrain"]:
    sample_complexity = {}
    sample_complexity["True"] = [[[10000000 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    sample_complexity["False"] = [[[10000000 for _ in range(4)] for _ in range(4)] for _ in range(4)]
    for CoT in ["True","False"]:
        for k in [1,2,3,4]:
            for num_layers in [1,2,3,4]:
                for num_heads in [1,2,3,4]:
                    for lr in [6e-5, 8e-5, 1e-4]:
                        path = f"model/binary_{n_samples}_{n_digits}_{k}_{CoT}_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
                        result = get_result(path)
                        if result != None:
                            sample_complexity[CoT][k-1][num_layers-1][num_heads-1] = min(sample_complexity[CoT][k-1][num_layers-1][num_heads-1], get_sample_complexity(result))
            print(k,sample_complexity[CoT][k-1])
    my_plot(sample_complexity)