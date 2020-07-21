import numpy as np
import os 

def show_all_files_in_directory(input_path):
    'This function reads the path of all files in directory input_path with extension extension'
    files_list=[]
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".npz"):
               files_list.append(os.path.join(path, file))
    return files_list



def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def open_npz_file(File):
    with np.load(File) as data:
        keys = data.files
        content = data[keys[0]]
    return content


def dataset_summary(path):
    all_modals = show_all_files_in_directory(path)
    print(all_modals)
    stats = {}
    for i in all_modals:
        name = i.split('/')[-1].split('.')[0] 
        name_shape = open_npz_file(i).shape
        stats[name] = name_shape
    print(stats)
