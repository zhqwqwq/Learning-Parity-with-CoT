import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib.ticker import LogLocator, FuncFormatter

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
                return result_step["epoch"] * 10000000
    return 10000000

num_layers = 1
num_heads = 1
n_digits = 30
n_samples = 10000000
total_samples = 10000000

for embedding in ["gpt2_tiny_wpetrain"]:
    sample_complexity = {}
    sample_complexity["True"] = [10000000 for i in range(4)]
    sample_complexity["False"] = [10000000 for i in range(4)]
    for CoT in ["True","False"]:
        for k in [1,2,3,4]:
            for lr in [6e-5, 8e-5, 1e-4]:
                path = f"model/binary_{n_samples}_{n_digits}_{k}_{CoT}_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
                result = get_result(path)
                if result != None:
                    print(CoT, k, lr,  get_sample_complexity(result))
                    sample_complexity[CoT][k-1] = min(sample_complexity[CoT][k-1], get_sample_complexity(result))
        print(sample_complexity[CoT])


    plt.figure(figsize=(6.5, 6))
    
    x = [1,2,3,4]
    plt.plot(x, sample_complexity["True"], label="with CoT", marker='o')
    plt.plot(x, sample_complexity["False"], label="without CoT", marker='x')
    plt.yscale('log')

    def y_formatter(y, pos):
        if y == 1e7:
            return '≥ 10⁷'
        else:
            return f'$10^{{{int(np.log10(y))}}}$'
    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_formatter))

    plt.xlabel("Number of Secret Variables $k$", fontsize=24)
    plt.ylabel("Sample Complexity", fontsize=24)
    plt.xticks([1, 2, 3, 4], fontsize=18)  
    plt.yticks(fontsize=18)  

    # Add legend
    plt.legend(fontsize=20)

    plt.grid(True, which="major", ls="--", linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("Figures/Figs/fig2.2.pdf")
    plt.savefig("Figures/Figs/fig2.2.svg")

