import numpy as np
import os
import re

def glob_re(directory, pattern):
    files = os.listdir(directory)
    files = filter(re.compile(pattern).match, files)
    files = [os.path.join(directory, f) for f in files]
    return sorted(files, reverse=True)

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def get_dataset_label(filename):
    dataset = filename.split('_clf')[0].split('_')[-1].split('.')[0]
    if dataset=='imagenet':
        return 'ImageNet'
    elif dataset=='unlabeled':
        return 'magnitudes'
    elif dataset=='None':
        return 'from scratch'
    return dataset

def get_finetuning_suffix(filename):
    ft = bool(int(filename.split('_ft')[1][0]))
    if ft:
        return 'w/ finetuning'
    return 'w/o finetuning'

def get_channels_label(filename):
    return ''.join(filter(str.isdigit, filename.split('/')[-1].split('_ft')[0]))[-2:].strip('0') + ' ch'
