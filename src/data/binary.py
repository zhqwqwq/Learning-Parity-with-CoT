import numpy as np
import torch
from torch.utils.data import Dataset
from .my_dataset import MyDataset
from tqdm import tqdm
import json
import os
from transformers import AutoTokenizer      
class BinaryDataset(MyDataset):
    def __init__(self, kwargs, generate = True):
        if('secret' not in kwargs):
            n_digit = kwargs['n_digit']
            n_secret = kwargs['n_secret']
            while True:
                secret = torch.zeros([n_digit])
                secret_idx = np.random.choice(n_digit, n_secret, replace=False)
                secret[secret_idx] = 1
                if secret[-1] != 1 or n_digit == n_secret: # We want the last bit not to be 1.
                    break
            kwargs['secret'] = secret
            kwargs['secret_idx'] = secret_idx
        self.auxiliary_level = kwargs.get('auxiliary_level', 1)
        self.use_long_cot = False
        self.position_sensitive = False
        super().__init__(kwargs, generate)

    def generate(self,n_samples):
        for i in tqdm(range(n_samples)):
            self._generate(is_fromlist = 'input_list' in self.kwargs, idx = i)
        
    def _generate(self, is_fromlist = False, idx = None):
        n_digit = self.n_digit
        if is_fromlist == True:
            q = []
            tmp = self.input_list[idx]
            for k in range(n_digit):
                q.append(tmp&1)
                tmp>>=1
            q = torch.tensor(q)
        else: # random
            q = torch.randint(0, 2, [n_digit])
        if(not self.position_sensitive):
            self.input_ids.append([_.item() for _ in list(q)])
        else:
            self.input_ids.append([_.item() + 2 * i for i, _ in enumerate(list(q))])
        self.labels.append([-100] * len(list(q)))
        self.auxiliary_labels.append([[2] for _ in range(self.auxiliary_level)])
        auxiliary_length = (self.n_secret + 2) // self.auxiliary_level
        cnt = 0
        for id in range(n_digit):
            if(self.use_long_cot or self.secret[id]):
                cnt += 1
                partial_y = (torch.sum(q[:id] * self.secret[:id]) % 2).int().item()
                if(self.use_cot):
                    self.input_ids[-1].append(partial_y)
                    self.labels[-1].append(partial_y)
                self.auxiliary_labels[-1][cnt // auxiliary_length].append(partial_y)
        for _ in range(self.auxiliary_level):
            self.auxiliary_labels[-1][_] = self.auxiliary_labels[-1][_] + [0] * ((auxiliary_length) - len(self.auxiliary_labels[-1][_]))
        y = (torch.sum(q * self.secret) % 2).int().item()
        self.input_ids[-1].append(y)
        self.labels[-1].append(y)
        self.attention_mask.append([1] * len(self.input_ids[-1]))
        self.auxiliary_inputs.append(self.auxiliary_labels[-1])
        self.data.append(q)

    def save(self, output_dir):
        super().save(output_dir)
        metadata = {
            'secret': [_.item() for _ in list(self.secret)]
        }
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_name(self):
        return f'binary_{self.n_samples}_{self.n_digit}_{self.n_secret}_{self.use_cot}_{self.use_long_cot}_{self.position_sensitive}'
