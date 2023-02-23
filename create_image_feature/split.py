from PIL import Image
import glob
import numpy as np
import heapq
from matplotlib import cm
import random
import os
from tqdm import tqdm
from shutil import copyfile


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False



def show_all_files_in_directory(input_path):
    'This function reads the path of all files in directory input_path'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".png"):
               files_list.append(os.path.join(path, file))
    return files_list


input_path = '/home/batool/beam_selection/image/crops/bus'
save_path = '/home/batool/beam_selection/image/data/'
number_of_files_in_directory = len(show_all_files_in_directory(input_path))
print('There are {} files in directory'.format(number_of_files_in_directory))


for experiment in tqdm(show_all_files_in_directory(input_path)):
        name = experiment.split('/')[-1]

        random_number = random.uniform(0, 1)
        if random_number<0.7:
            copyfile(experiment, save_path+'train/bus/'+name)
        elif 0.7<random_number<0.85:
            copyfile(experiment, save_path+'validation/bus/'+name)
        elif 0.85<random_number:
            copyfile(experiment, save_path+'test/bus/'+name)
