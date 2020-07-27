from __future__ import print_function

import argparse
import numpy as np
import os

from stats import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'mmwave framework',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--id_gpu', default=0, type=int, help='which gpu to use.')
    parser.add_argument('--base_path', default='/home/batool/beam_selection/baseline_data/', help='base path of workspace directory')
    parser.add_argument('--save_path', default='/home/batool/beam_selection/baseline_data/', help='base path of workspace directory')
    parser.add_argument('--k', default=1, type=int, help='top k')

    args = parser.parse_args()

    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

######################################Initilzation
    print("*************Dataset Shapes***************")
    N_tr = 32
    N_rx = 8
    dataset_summary(args.base_path)
    print("*************Mapping***************")
    All_possible_pairs = [(i,j) for i in range(N_tr) for j in range(N_rx)]
    mapping = labels_to_categorical(All_possible_pairs)
    print("Mapping to unique lables",mapping)
######################################
    print("*************Class diversity***************")
    Class_diversity = stats_beams(args.base_path+"beam_output", mapping, N_tr = 32, N_rx = 8, k =args.k)
    print(Class_diversity)
