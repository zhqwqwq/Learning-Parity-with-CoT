from seaborn import heatmap
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import random
import json
import os

fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
n_digits = 20

(embedding,k) = ("gpt2_tiny_wpetrain",12)
for j in range(3):
    (n_samples,total_samples) = [(10000,10000000),(100000,10000000),(1000000,10000000)][j]
    accuracy = np.zeros((6,6))
    for lr in [6e-5,8e-5,1e-4]:
        for idx_layers in range(6):
            for idx_heads in range(6):
                num_layers = [1,2,3,4,6,8][idx_layers]
                num_heads = [1,2,3,4,6,8][idx_heads]
                path = f"model/binary_{n_samples}_{n_digits}_{k}_False_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        path = item_path
                path = os.path.join(path,"trainer_state.json")
                with open(path, 'r') as f:
                    result = json.load(f)
                accuracy[idx_layers][idx_heads] = max(accuracy[idx_layers][idx_heads],result["log_history"][-1]["eval_exact_match"])
    heatmap(accuracy, ax=axs[j], vmin=0.4, vmax=1, cmap="YlGnBu", cbar=False)
    axs[j].set_title(f'{n_samples} samples × {total_samples//n_samples} epochs', fontsize = 20)
    axs[j].set_xticklabels([1, 2, 3, 4,6,8],fontsize=22)
    axs[j].set_yticklabels([1, 2, 3, 4,6,8],fontsize=22)
    axs[j].set_xlabel('Head Number', fontsize = 24)
    axs[j].set_ylabel('Layer Number',  fontsize = 24)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
cbar = fig.colorbar(axs[0].collections[0], cax=cbar_ax,)
cbar.ax.tick_params(labelsize=18) 
plt.tight_layout(rect=[0, 0, 0.9, 0.9], pad = 1) 
fig.suptitle('accuracy', fontsize=26, y=1)
plt.show()
plt.savefig(f'Figures/Figs/fig9.pdf')