from __future__ import print_function

import argparse
import pickle
import numpy as np
import os

from PIL import Image
from TrainTest import TrainTest,get_models,Load_Entire_Image,show_all_files_in_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'mmwave framework',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--id_gpu', default=2, type=int, help='which gpu to use.')

    parser.add_argument('--base_path', default='/home/batool/beam_selection/image/', type=str, help='base path of whole wall including images and pickle file')

    parser.add_argument('-bs', '--batch_size', type=int, default=256,help='Batch size')
    parser.add_argument('--epochs', type=int, default=4,help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,help='Number of epochs')
    parser.add_argument('--num', type=int, default=4,help='number of classes')

    parser.add_argument('--restore_models', default=True,help='restore model or not')

    parser.add_argument('--model_path', default='/home/batool/beam_selection/image/',help='path of wall model')
    parser.add_argument('--model_json', default='/home/batool/beam_selection/image/model.json', help = "restore json files from here")
    parser.add_argument('--model_weight', default='/home/batool/beam_selection/image/model_weights.hdf5',help='restore weights from')

    parser.add_argument('--stride', type=int,default=5,help='sweeping stride')
    parser.add_argument('--window', type=int,default=40,help='Window size')
    parser.add_argument('--channels', type=int,default=3,help='Window size')

    ################## Entire Image Parametres #####################
    parser.add_argument('--path_of_entire_image', default='/home/batool/beam_selection/image/entire_images/',help='path of crops of entire image')


    args = parser.parse_args()

    if args.id_gpu >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # The GPU id to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

##################################################################Binary Classifier of Wall
    pipeline = TrainTest(base_path=args.base_path, save_path=args.base_path)
    print('**********************Train Wall Camera Binary Classifier**********************')

    if not args.restore_models:
        print('**********************Add new model**********************')

        model = get_models('seperate',inputshape=(args.window,args.window,args.channels), classes=args.num, lr=args.lr)
        model.summary()

        pipeline.add_model(classes=args.num, model_flag='seperate', model=model, model_path=args.model_path)


    elif  args.restore_models:
        print('***************Adding Existing Model and weights***************')
        model = pipeline.load_model_structure(args.num, args.model_json)
        model.summary()

        pipeline.add_model(classes=args.num, model_flag='seperate', model=model,model_path=args.model_path)
        pipeline.load_weights(args.model_weight)
        print("***************Pretrained model weights are loaded***************")

    if False:
        print("***************Training***************")

        pipeline.train_model(data_path = args.base_path+'data', batch_size=args.batch_size, window=args.window, lr=args.lr, epochs=args.epochs, model_path = args.model_path)

    if True:

        print("***************Testing***************")

        pipeline.test_model(data_path = args.base_path+'data', batch_size=args.batch_size, window=args.window, lr=args.lr, epochs=args.epochs, model_path = args.model_path)

##################################################################Entire Image Part
    print('***************Load Entire Image***************')

    # prediction on entire images+ effected
    print(show_all_files_in_directory(args.path_of_entire_image))

    pipeline.predict_on_crops(entire_images_path = show_all_files_in_directory(args.path_of_entire_image), window = args.window, stride = args.stride)

