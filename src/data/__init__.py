from .binary import BinaryDataset
import torch
import os

dataset_type_list = ['BinaryDataset']
def load_dataset(input_dir, dataset_type = 'BinaryDataset'):
    dataset = eval(dataset_type)(torch.load(os.path.join(input_dir, 'kwargs.pt')), generate = False)
    dataset.load(input_dir)
    return dataset
