import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import json
import torch
from math import log
import numpy as np
import re

def compute_entropy(attention_vector):
    entropy = -torch.sum(attention_vector * torch.log(attention_vector + 1e-9), dim=-1)
    return entropy

def compute_min_attention_entropy_over_token(attentions, input_token_num = 0):
    layer_head_entropies = []
    for layer_attention_ in attentions:  
        layer_attention = layer_attention_[0] 
        head_entropies = []
        for head_index in range(layer_attention.shape[0]): 
            head_attention = layer_attention[head_index, :, :] 
            log_tensor = torch.log(torch.arange(2, head_attention.shape[0] + 1, dtype=torch.float)).unsqueeze(0).to(head_attention.device)
            entropy = compute_entropy(head_attention) 
            normalized_entropy = entropy[1:] = entropy[1:] / log_tensor
            min_entropy_over_token = normalized_entropy[:,input_token_num:].min(dim=1).values.item()
            head_entropies.append(min_entropy_over_token)
        layer_head_entropies.append(head_entropies)
    return layer_head_entropies

def compute_average_attention_entropy_over_token(attentions, input_token_num = 0):
    layer_head_entropies = []
    for layer_attention_ in attentions:
        layer_attention = layer_attention_[0] 
        head_entropies = []
        for head_index in range(layer_attention.shape[0]): 
            head_attention = layer_attention[head_index, :, :] 
            log_tensor = torch.log(torch.arange(2, head_attention.shape[0] + 1, dtype=torch.float)).unsqueeze(0).to(head_attention.device)
            entropy = compute_entropy(head_attention) 
            normalized_entropy = entropy[1:] = entropy[1:] / log_tensor
            average_entropy_over_token = torch.mean(normalized_entropy[:,input_token_num:],dim = 1).item()
            head_entropies.append(average_entropy_over_token)
        layer_head_entropies.append(head_entropies)
    return layer_head_entropies


device = "cuda"
math_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-Math-7B")
math_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-Math-7B")
math_model.to(device)
math_model.eval()
math_model.config.output_attentions = True

qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B")
qwen_model.to(device)
qwen_model.eval()
qwen_model.config.output_attentions = True

# Load dataset
gsm8k_dataset = pd.read_parquet('gsm8k/test-00000-of-00001.parquet').to_dict(orient='records')
for sample in gsm8k_dataset:
    sample['CoT_text'] = sample['question'] + ' '+sample['answer']
    match = re.search(r'(\d+)\s*$', sample['answer'])
    sample['noCoT_text'] = sample['question'] + f" The answer is {match.group(1)}."
    print(sample['noCoT_text'])


results = []
for sample in gsm8k_dataset:
    text = sample["CoT_text"]
    inputs = math_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print(inputs)
    output_ids = math_model(**inputs)
    print(output_ids.keys())

    attention = output_ids.attentions
    min_normalized_entropy = compute_min_attention_entropy_over_token(attention, sample["math_input_token_num"])

    print(min_normalized_entropy)
    results.append({
        "test_text": text,
        "attention_maps": min_normalized_entropy  
    })

path = "Figures/Fig6-7/math_CoT.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

results = []
for sample in gsm8k_dataset:
    text = sample["CoT_text"]
    inputs = qwen_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print(inputs)
    output_ids = qwen_model(**inputs)
    print(output_ids.keys())

    attention = output_ids.attentions
    min_normalized_entropy = compute_min_attention_entropy_over_token(attention, sample["math_input_token_num"])

    print(min_normalized_entropy)
    results.append({
        "test_text": text,
        "attention_maps": min_normalized_entropy  
    })

path = "Figures/Fig6-7/qwen_CoT.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
    
results = []
for sample in gsm8k_dataset:
    text = sample["noCoT_text"]
    inputs = qwen_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    print(inputs)
    output_ids = qwen_model(**inputs)
    print(output_ids.keys())

    attention = output_ids.attentions
    min_normalized_entropy = compute_min_attention_entropy_over_token(attention, sample["math_input_token_num"])

    print(min_normalized_entropy)
    results.append({
        "test_text": text,
        "attention_maps": min_normalized_entropy  
    })

path = "Figures/Fig6-7/qwen_noCoT.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)