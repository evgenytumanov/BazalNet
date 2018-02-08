import os
from os.path import join as opj
import numpy as np
import glob
import re
from sklearn.model_selection import train_test_split

def iterate_minibatches_global(inputs, targets, batchsize, start_it, iters):
    len_ = len(inputs)
    assert len_ == len(targets)
    indices = np.arange(len(inputs))

    for it in range(start_it, iters):
        if it % len_ == 0:
            np.random.shuffle(indices)

        if it % len_ < len_ - batchsize + 1:
            excerpt = indices[it % len_:it % len_ + batchsize]
            yield inputs[excerpt], targets[excerpt]

def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

