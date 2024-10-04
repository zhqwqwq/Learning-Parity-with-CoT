import matplotlib.pyplot as plt
import numpy as np
import json



num_layer = 28
num_head = 28

for text_type in ['notonlyanswer','onlyanswer']:
    for average_type in ['min', 'average']:
        with open(f'Experiment_Section/GSM8K/Additional_Experiment/qwen_CoT_{text_type}_{average_type}.json', 'r', encoding='utf-8') as file:
            qwen_CoT = json.load(file)  
        with open(f'Experiment_Section/GSM8K/Additional_Experiment/qwen_noCoT_{text_type}_{average_type}.json', 'r', encoding='utf-8') as file:
            qwen_noCoT = json.load(file)  
        with open(f'Experiment_Section/GSM8K/Additional_Experiment/math_CoT_{text_type}_{average_type}.json', 'r', encoding='utf-8') as file:
            math_CoT = json.load(file)
        total_data = len(qwen_CoT)
        print(total_data)
        sum_qwenCoT_entropy = [[0 for _ in range(num_head)] for _ in range(num_layer)]
        sum_qwennoCoT_entropy = [[0 for _ in range(num_head)] for _ in range(num_layer)]
        sum_mathCoT_entropy  = [[0 for _ in range(num_head)] for _ in range(num_layer)]
        for id in range(total_data):
            for i in range(num_layer):
                for j in range(num_head):
                    sum_qwenCoT_entropy[i][j] += qwen_CoT[id]["attention_maps"][i][j]
                    sum_qwennoCoT_entropy[i][j] += qwen_noCoT[id]["attention_maps"][i][j]
                    sum_mathCoT_entropy[i][j] += math_CoT[id]["attention_maps"][i][j]
        qwenCoT_entropy = [[0 for _ in range(num_head)] for _ in range(num_layer)]
        qwennoCoT_entropy = [[0 for _ in range(num_head)] for _ in range(num_layer)]
        mathCoT_entropy = [[0 for _ in range(num_head)] for _ in range(num_layer)]
        for i in range(num_layer):
            for j in range(num_head):
                qwenCoT_entropy[i][j] = sum_qwenCoT_entropy[i][j]/total_data
                qwennoCoT_entropy[i][j] = sum_qwennoCoT_entropy[i][j]/total_data
                mathCoT_entropy[i][j] = sum_mathCoT_entropy[i][j]/total_data
        for i in range(num_layer):
            qwenCoT_entropy[i].sort()
            qwennoCoT_entropy[i].sort()
            mathCoT_entropy[i].sort()

        indices = np.arange(1,len(qwennoCoT_entropy)+1)

        bar_width = 1
        fig, axs = plt.subplots(1, 4, figsize=(31, 6.5))  # 设置图表大小

        for i in range (4):
            layer = [0,10,20,27][i]
            ax = axs[i]
            
            ax.bar(indices, qwennoCoT_entropy[layer], width=bar_width, label='Qwen2-7B+\nNo CoT', color='#f6bebf')
            ax.bar(indices, qwenCoT_entropy[layer], width=bar_width, label='Qwen2-7B+\nWith CoT', color='#4187A2')
            ax.bar(indices, mathCoT_entropy[layer], width=bar_width, label='Qwen2-Math-7B+\nWith CoT', color='#C6DBAD')
            ax.plot(indices, mathCoT_entropy[layer], marker='o', color='#59ac50', linestyle='-', linewidth=2, markersize=8)
            ax.plot(indices, qwenCoT_entropy[layer], marker='D', color='#244c7e', linestyle='-', linewidth=2, markersize=8)
            ax.plot(indices, qwennoCoT_entropy[layer], marker='x', color='#C95762', linestyle='-', linewidth=2, markersize=8)

            ax.set_title(f'Layer {layer+1}',fontsize=36)
            ax.set_xlabel('head',fontsize=36)
            if i == 0:
                ax.set_ylabel('Normalized Attention Entropy',fontsize=28)
            if layer == 27:
                ax.set_yticks([0,0.1])
            ax.tick_params(axis='both', which='major', labelsize=30)  
            ax.set_xticks([1,28])
        handles, labels = axs[0].get_legend_handles_labels() 
        fig.legend(handles, labels, loc='center left', fontsize=30, bbox_to_anchor=(0.83, 0.5), frameon=False)  
        plt.subplots_adjust(wspace=0.1) 
        plt.tight_layout(rect=[0, 0, 0.83, 1])
        plt.show()
        plt.savefig(f'Figures/Figs/fig6.pdf')