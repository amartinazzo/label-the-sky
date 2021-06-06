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
