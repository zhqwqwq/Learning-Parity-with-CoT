import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer      
class MyDataset(Dataset):
    def __init__(self, kwargs, generate = True):
        assert 'n_samples' in kwargs, 'n_samples is required'
        self.init_from_args(kwargs)
        self.input_ids = []
        self.attention_mask = []
        self.data = []
        self.labels = []
        self.auxiliary_inputs = []
        self.auxiliary_labels = []
        if(generate):
            self.generate(self.n_samples)       
    def init_from_args(self, kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.kwargs = kwargs
        if 'tokenizer_name' in kwargs:
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer_name'])   
            self.tokenizer.pad_token = self.tokenizer.eos_token   
            self.tokenizer.padding_side = "left"
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if(len(self.auxiliary_labels) > 0):
            return {'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx], 
                    'data':  self.data[idx],
                    'auxiliary_labels': self.auxiliary_labels[idx], 
                    'auxiliary_inputs': self.auxiliary_inputs[idx]}
        else:        
            return {'input_ids': self.input_ids[idx],
                    'attention_mask': self.attention_mask[idx],
                    'labels': self.labels[idx], 
                    'data':  self.data[idx],}
    def generate(self, n_samples):
        for i in tqdm(range(n_samples)):
            self._generate()
    def _generate(self):
        assert NotImplementedError
            
    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save({
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'labels': self.labels,
            'data': self.data,
            'auxiliary_labels': self.auxiliary_labels,
            'auxiliary_inputs': self.auxiliary_inputs,
        }, f'{output_dir}/tensor_data.pt')
        torch.save(self.kwargs, f'{output_dir}/kwargs.pt')
            
    
    def load(self, input_dir):
        kwargs = torch.load(f'{input_dir}/kwargs.pt')
        self.init_from_args(kwargs)
        tensor_data = torch.load(f'{input_dir}/tensor_data.pt')
        self.input_ids = tensor_data['input_ids']
        self.attention_mask = tensor_data['attention_mask']
        self.labels = tensor_data['labels']
        self.data = tensor_data['data']
        if('auxiliary_labels' in tensor_data):
            self.auxiliary_labels = tensor_data['auxiliary_labels']
            self.auxiliary_inputs = tensor_data['auxiliary_inputs']
    def get_name(self):
        assert NotImplementedError
        
    
        
