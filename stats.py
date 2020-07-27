import os
import numpy as np
from  more_itertools import unique_everseen

def show_all_files_in_directory(input_path):
    """This function reads the path of all files in directory input_path with extension .npz"""
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
    """Assuming to have a single key, as our dataset"""
    with np.load(File) as data:
        keys = data.files
        content = data[keys[0]]
    return content


def dataset_summary(path):
    """Returns shape of dataset"""
    all_modals = show_all_files_in_directory(path)
    stats = {}
    for i in all_modals:
        name = i.split('/')[-1].split('.')[0]
        name_shape = open_npz_file(i).shape
        stats[name] = name_shape
    print("Dataset shapes",stats)


def labels_to_categorical(labels):
    """
    Convert pair of labels to unique indexes. That output is
    a dictionary as {(0, 0):0, (0, 1):1,...,(8,32):255}
    """
    dis=list(unique_everseen(labels))
    mapping=[(element,dis.index(element)) for element in dis]
    return dict(mapping)


def top_K_beams(beams,k,mapping,N_tr=32,N_rx=8):
    # The output is (case, [beam1,beam2,...])

    Top_K=[]
    for count,val in enumerate(beams):   # beams is 9..*32*8
        Linear = val.reshape(val.shape[0]*val.shape[1])
        Sorted = Linear.argsort()[-k:][::-1]
        Indexes = [(i-N_tr*int(i/N_tr),int(i/N_tr)) for i in Sorted]
        Map = [mapping[i] for i in Indexes]
        Top_K.append((count,Map))
    return Top_K
##################################For beams

def stats_beams(beam_path, mapping ,N_tr = 32, N_rx = 8, k =2):

    Train_Val = []
    for beam_path in show_all_files_in_directory(beam_path):
        my_dict = {}
        beams =open_npz_file(beam_path)
        Top = top_K_beams(beams,k,mapping)   #(case1,[b1,b2]),(case2,[b1,b2])

        List_of_top = []
        for t in Top:
            List_of_top.extend(t[1][:])

        my_dict = {i:List_of_top.count(i) for i in List_of_top}
        name = beam_path.split('/')[-1].split('.')[0]
        Train_Val.append((name,my_dict))
    return Train_Val
