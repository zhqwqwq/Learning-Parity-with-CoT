import os
from data import *
import numpy as np
import argparse
import copy
import random

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default = '', type = str)
    args = parser.parse_args()
    config = eval(open(args.config_path, 'r').read())
    assert config['dataset_type'] == 'BinaryDataset' # only for Binary now.
    n_samples = config['n_samples']
    n_digit = config['n_digit']

    
    if n_digit <= 20:
        input_list = list(range(1<<n_digit))
        random.shuffle(input_list)
    elif n_digit <= 50:
        input_list = random.sample(range(0, 1<<n_digit), n_samples + config['val_samples'])
    else:
        input_list = [random.randint(0,2**n_digit-1) for _ in range(n_samples + config['val_samples'])]

    dataset_type = eval(config['dataset_type'])
    assert n_samples + config['val_samples'] <= (1<<n_digit)
    config['input_list'] = input_list[:n_samples]
    train_dataset = dataset_type(config) # change to a dictionary.
    val_config = copy.deepcopy(train_dataset.kwargs)
    val_config['n_samples'] = config['val_samples']
    val_config['input_list'] = input_list[n_samples:n_samples+config['val_samples']]
    val_dataset = dataset_type(val_config)
    name = train_dataset.get_name()
    print(train_dataset[0])
    print('Input:')
    print((train_dataset.input_ids[0]))
    print((val_dataset.input_ids[0]))
    print(train_dataset.secret)
    PATH = "data"
    os.makedirs(f'{PATH}/Nonintersect_Binary/{name}', exist_ok=True)
    train_dataset.save(f'{PATH}/Nonintersect_Binary/{name}/')
    val_dataset.save(f'{PATH}/Nonintersect_Binary/{name}/val')