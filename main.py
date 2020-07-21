from __future__ import print_function

import argparse
import numpy as np
import os

from TrainTest import dataset_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'mmwave framework',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--id_gpu', default=2, type=int, help='which gpu to use.')
    parser.add_argument('--base_path', default='/home/batool/beam_selectiom/baseline_data', help='base path of workspace directory')

    args = parser.parse_args()

    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)



    dataset_summary(args.base_path)
