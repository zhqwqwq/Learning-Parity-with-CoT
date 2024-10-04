import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os

def get_result(path):
    if os.path.isdir(path) == False:
        return None
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

def get_sample_complexity(result, thre):
    for index, result_step in enumerate(result["log_history"]):
        if index % 2 == 1:
            if result_step["eval_exact_match"] >= thre:
                print(result_step["epoch"] * 1000000)
                return result_step["epoch"] * 1000000
    return 1000000

num_layers = 1
num_heads = 1
n_digits = 100
n_samples = 1000000
total_samples = 1000000
CoT = "True"

for embedding in ["gpt2_tiny_wpetrain"]:
    sample_complexity = [1000000 for i in range(9)]
    for idx in range(9):
        k = [20, 30, 40, 50, 60, 70, 80, 90, 100][idx]
        for lr in [6e-5, 8e-5, 1e-4]:
        # for lr in [1e-4]:
            path = f"model/binary_{n_samples}_{n_digits}_{k}_{CoT}_False_False_{total_samples}_LR={lr}_WD=0.0_1GPU*512Batch_{embedding}.py_#layer={num_layers}_#head={num_heads}"
            result = get_result(path)
            if result != None:
                sample_complexity[idx] = min(sample_complexity[idx], get_sample_complexity(result,0.995))
        print(sample_complexity)

    x = [20, 30, 40, 50, 60, 70, 80, 90, 100]

    plt.figure(figsize=(6, 5.5))

    plt.plot(x, sample_complexity, marker='x', label="with CoT")

    def y_format_func(value, tick_number):
        return f'{value / 1e5:.1f}'

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(y_format_func))

    plt.gca().set_ylabel("Sample Complexity ($\\times10^5$)", fontsize=24)

    plt.xlabel("Number of Secret Variables $k$", fontsize=24)
    plt.xticks([20, 30, 40, 50, 60, 70, 80, 90, 100], fontsize=18)
    plt.yticks(fontsize=18)

    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.show()
    plt.savefig("Figures/Figs/fig3.1.pdf")
    plt.savefig("Figures/Figs/fig3.1.svg")

