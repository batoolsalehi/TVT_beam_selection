from __future__ import division

import csv
import numpy as np
from tqdm import tqdm
import random
import os


def open_npz(path,key):
    data = np.load(path)[key]
    return data


data_folder = '/home/batool/beam_selection/baseline_data/'
save_folder = '/home/batool/beam_selection/baseline_code_modified/jen/'


output_train_file = data_folder+'beam_output/beams_output_train.npz'
output_validation_file = data_folder+'beam_output/beams_output_validation.npz'


print("Reading outputs...")
output_cache_file_tr = np.load(output_train_file)
ytrain = output_cache_file_tr['output_classification']

ytrain32 = ytrain.sum(axis=1)
ytrain8 = ytrain.sum(axis=2)
print(ytrain32.shape,ytrain8.shape)

np.savez_compressed(save_folder+'train32.npz', train32=ytrain32)
np.savez_compressed(save_folder+'train8.npz', train8=ytrain8)



output_cache_file_val = np.load(output_validation_file)
yval = output_cache_file_val['output_classification']

yval32 = yval.sum(axis=1)
yval8 = yval.sum(axis=2)

np.savez_compressed(save_folder+'val32.npz', val32=yval32)
np.savez_compressed(save_folder+'val8.npz', val8=yval8)

print(yval32.shape,yval8.shape)

