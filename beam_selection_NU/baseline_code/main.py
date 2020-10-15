########################################################
#Project name: ITU beam selection challenge
#Authors: NU Huskies team
#Date: 15/Oct/2020
########################################################
from __future__ import division

import os
import csv
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import random
from time import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model, load_model
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization

import sklearn
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest

from ModelHandler import add_model,load_model_structure, ModelHandler
from custom_metrics import *
############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
tf.set_random_seed(seed)
#tf.random_set_seed()
np.random.seed(seed)
random.seed(seed)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def open_npz(path,key):
    data = np.load(path)[key]
    return data

def save_npz(path,train_name,train_data,val_name,val_data):
    check_and_create(path)
    np.savez_compressed(path+train_name, train=train_data)
    np.savez_compressed(path+val_name, val=val_data)


def beamsLogScale(y,thresholdBelowMax):
        y_shape = y.shape   # shape is (#,256)

        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs + 1e-30)
            minValue = np.amax(logOut) - thresholdBelowMax
            zeroedValueIndices = logOut < minValue
            thisOutputs[zeroedValueIndices]=0
            thisOutputs = thisOutputs / sum(thisOutputs)
            y[i,:] = thisOutputs
        return y

def getBeamOutput(output_file):
    thresholdBelowMax = 6
    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]

    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y = beamsLogScale(y,thresholdBelowMax)

    return y,num_classes

def custom_label(output_file, strategy='one_hot' ):
    'This function generates the labels based on input strategies, one hot, reg'

    print("Reading beam outputs...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)

    yMatrixShape = yMatrix.shape
    num_classes = yMatrix.shape[1] * yMatrix.shape[2]
    y = yMatrix.reshape(yMatrix.shape[0],num_classes)
    y_shape = y.shape

    if strategy == 'one_hot':
        k = 1           # For one hot encoding we need the best one
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs)
            max_index = logOut.argsort()[-k:][::-1]
            y[i,:] = 0
            y[i,max_index] = 1

    elif strategy == 'reg':
        for i in range(0,y_shape[0]):
            thisOutputs = y[i,:]
            logOut = 20*np.log10(thisOutputs)
            y[i,:] = logOut
    else:
        print('Invalid strategy')
    return y,num_classes


def balance_data(beams,modal,variance,dim):
    'This function balances the dataset by generating multiple copies of classes with low frequency'

    Best_beam=[]    # a list of 9234 elements(0,1,...,256) with the index of best beam
    k = 1           # Augment on best beam
    for count,val in enumerate(beams):   # beams is 9234*256
        Index = val.argsort()[-k:][::-1]
        Best_beam.append(Index[0])

    Apperance = {i:Best_beam.count(i) for i in Best_beam}   #Selected beam:  count apperance of diffrent classes
    print(Apperance)
    Max_apperance = max(Apperance.values())                 # Class with highest apperance

    for i in tqdm(Apperance.keys()):
        ind = [ind for ind, value in enumerate(Best_beam) if value == i]    # Find elements which are equal to i
        randperm = np.random.RandomState(seed).permutation(int(Max_apperance-Apperance[i]))%len(ind)

        ADD_beam = np.empty((len(randperm), 256))
        extension = (len(randperm),)+dim
        ADD_modal = np.empty(extension)
        print('shapes',ADD_beam.shape, ADD_modal.shape)

        for couter,v in enumerate(randperm):
            ADD_beam[couter,:] = beams[ind[v]]
            ADD_modal[couter,:] = modal[ind[v]]+variance*np.random.rand(*dim)
        beams = np.concatenate((beams, ADD_beam), axis=0)
        modal = np.concatenate((modal, ADD_modal), axis=0)

    return beams, modal



parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--test_data_folder', help='Location of the test data directory', type=str)

parser.add_argument('--epochs', default=10, type = int, help='Specify the epochs to train')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)

parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--Aug', type=str2bool, help='Do Augmentaion to balance the dataset or not', default=False)
parser.add_argument('--augmented_folder', help='Location of the augmeneted data', type=str, default='G:/Beam_Selection_ITU/beam_selection/baseline_code_modified/aug_data/')

parser.add_argument('--fusion_architecture', type=str ,default='mlp', help='Whether to use convolution in fusion network architecture',choices=['mlp','cnn'])
parser.add_argument('--img_version', type=str, help='Which version of image folder to use', default='')

parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = '/home/batool/beam_selection_NU/baseline_code/model_folder/')

parser.add_argument('--image_feature_to_use', type=str ,default='v1', help='feature images to use',choices=['v1','v2','custom'])



args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
start = time()

check_and_create(args.model_folder)
###############################################################################
# Outputs
###############################################################################
# Read files
output_train_file = args.data_folder+'beam_output/beams_output_train.npz'
output_validation_file = args.data_folder+'beam_output/beams_output_validation.npz'
output_test_file = args.test_data_folder+'beam_output/beams_output_test.npz'


if args.strategy == 'default':
    y_train,num_classes = getBeamOutput(output_train_file)
    y_validation, _ = getBeamOutput(output_validation_file)
    y_test, _ = getBeamOutput(output_test_file)
elif args.strategy == 'one_hot' or args.strategy == 'reg':
    y_train,num_classes = custom_label(output_train_file,args.strategy)
    y_validation, _ = custom_label(output_validation_file,args.strategy)
    y_test, _ = custom_label(output_test_file,args.strategy)

else:
    print('invalid labeling strategy')

###############################################################################
# Inputs
###############################################################################
Initial_labels_train = y_train         # these are same for all modalities
Initial_labels_val = y_validation

if 'coord' in args.input:
    #train
    X_coord_train = open_npz(args.data_folder+'coord_input/coord_train.npz','coordinates')
    #validation
    X_coord_validation = open_npz(args.data_folder+'coord_input/coord_validation.npz','coordinates')
    #test
    X_coord_test = open_npz(args.test_data_folder + 'coord_input/coord_test.npz', 'coordinates')
    coord_train_input_shape = X_coord_train.shape

    if args.Aug:
        try:
            #train
            X_coord_train = open_npz(args.augmented_folder+'coord_input/coord_train.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            X_coord_validation = open_npz(args.augmented_folder+'coord_input/coord_validation.npz','val')
            y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')
        except:
            print('****************Augment coordinates****************')
            y_train, X_coord_train = balance_data(Initial_labels_train,X_coord_train,0.001,(2,))
            # y_validation, X_coord_validation = balance_data(Initial_labels_val,X_coord_validation,0.001,(2,))
            save_npz(args.augmented_folder+'coord_input/','coord_train.npz',X_coord_train,'coord_validation.npz',X_coord_validation)
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)
    ### For convolutional input
    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))
    print(X_coord_train.shape)


if 'img' in args.input:
    ###############################################################################
    resizeFac = 20 # Resize Factor
    nCh = 1 # The number of channels of the image
    if args.image_feature_to_use == 'v1':
        folder = 'image_input'
    elif args.image_feature_to_use == 'v2':
        folder =  'image_v2_input'
    elif args.image_feature_to_use == 'custom':
        folder = 'image_custom_input'

    #train
    X_img_train = open_npz(args.data_folder+folder +'/img_input_train_'+str(resizeFac)+'.npz','inputs')
    #validation
    X_img_validation = open_npz(args.data_folder+folder+'/img_input_validation_'+str(resizeFac)+'.npz','inputs')
    #test
    X_img_test = open_npz(args.test_data_folder+folder+'/img_input_test_' + str(resizeFac) + '.npz','inputs')
    img_train_input_shape = X_img_train.shape
    print('shape first',X_img_test.shape)

    if args.Aug:
        try:
            print('****************Load augmented data****************')
            #train
            X_img_train = open_npz(args.augmented_folder+'image_input/img_input_train_20.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            X_img_validation = open_npz(args.augmented_folder+'image_input/img_input_validation_20.npz','val')
            y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')

        except:
            print('****************Augment Image****************')
            y_train, X_img_train = balance_data(Initial_labels_train,X_img_train,0.001,(48, 81, 1))
            # y_validation, X_img_validation = balance_data(Initial_labels_val,X_img_validation,0.001,(48, 81, 1))
            save_npz(args.augmented_folder+'image_input/','img_input_train_20.npz',X_img_train,'img_input_validation_20.npz',X_img_validation)
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

    if args.image_feature_to_use == 'v1' or args.image_feature_to_use =='v2':
        print('********************Normalize image********************')
        X_img_train = X_img_train.astype('float32') / 255
        X_img_validation = X_img_validation.astype('float32') / 255
        X_img_test = X_img_test.astype('float32') / 255
    elif args.image_feature_to_use == 'custom':
        print('********************Reshape images for convolutional********************')
        X_img_train = X_img_train.reshape((X_img_train.shape[0], X_img_train.shape[1], X_img_train.shape[2],1))
        X_img_validation = X_img_validation.reshape((X_img_validation.shape[0], X_img_validation.shape[1],X_img_validation.shape[2], 1))
        X_img_test = X_img_test.reshape((X_img_test.shape[0], X_img_test.shape[1], X_img_test.shape[2],1))


if 'lidar' in args.input:
    ###############################################################################
    #train
    X_lidar_train = open_npz(args.data_folder+'lidar_input/lidar_train.npz','input')
    #validation
    X_lidar_validation = open_npz(args.data_folder+'lidar_input/lidar_validation.npz','input')
    #test
    X_lidar_test = open_npz(args.test_data_folder + 'lidar_input/lidar_test.npz', 'input')
    lidar_train_input_shape = X_lidar_train.shape

    if args.Aug:
        try:
            X_lidar_train = open_npz(args.augmented_folder+'lidar_input/lidar_train.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            X_lidar_validation = open_npz(args.augmented_folder+'lidar_input/lidar_validation.npz','val')
            y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')
        except:
            print('****************Augment Lidar****************')
            y_train, X_lidar_train = balance_data(Initial_labels_train,X_lidar_train,0.001,(20, 200, 10))
            # y_validation, X_lidar_validation = balance_data(Initial_labels_val,X_lidar_validation,0.001,(20, 200, 10))
            save_npz(args.augmented_folder+'lidar_input/','lidar_train.npz',X_lidar_train,'lidar_validation.npz',X_lidar_validation)
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

##############################################################################
# Model configuration
##############################################################################
#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)
fusion = False if len(args.input) == 1 else True

modelHand = ModelHandler()
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

if 'coord' in args.input:
    if args.restore_models:
        coord_model = load_model_structure(args.model_folder+'coord_model.json')
        coord_model.load_weights(args.model_folder + 'best_weights.coord.h5', by_name=True)
    else:
        coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete',args.strategy, fusion)
        add_model('coord',coord_model,args.model_folder)


if 'img' in args.input:

    if args.image_feature_to_use == 'v1' or args.image_feature_to_use =='v2':
        model_type = 'light_image_v1_v2'
    elif args.image_feature_to_use == 'custom':
        model_type = 'light_image_custom'

    if args.restore_models:
        img_model = load_model_structure(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json')
        img_model.load_weights(args.model_folder + 'best_weights.img_'+args.image_feature_to_use+'.h5', by_name=True)
    else:
        img_model = modelHand.createArchitecture(model_type,num_classes,[img_train_input_shape[1],img_train_input_shape[2],1],'complete',args.strategy,fusion)
        add_model('image_'+args.image_feature_to_use,img_model,args.model_folder)

if 'lidar' in args.input:
    if args.restore_models:
        lidar_model = load_model_structure(args.model_folder+'lidar_model.json')
        lidar_model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)
    else:
        lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy, fusion)
        add_model('lidar',lidar_model,args.model_folder)

###############################################################################
# Fusion
###############################################################################

if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        x_train = [X_lidar_train, X_coord_train]
        x_validation = [X_lidar_validation, X_coord_validation]
        x_test = [X_lidar_test, X_coord_test]

        combined_model = concatenate([lidar_model.output, coord_model.output],name = 'cont_fusion_coord_lidar')
        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),name = 'dense1_fusion_coord_lidar')(combined_model)
        z = Dropout(0.5,name = 'drop_fusion_coord_lidar')(z)
        z = Dense(num_classes, activation="softmax", use_bias=True,name = 'dense2_fusion_coord_lidar')(z)
        model = Model(inputs=[lidar_model.input, coord_model.input], outputs=z)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_lidar.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

        print(hist.history.keys())
        print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])

        print('***************Testing the model************')
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)

    elif 'coord' in args.input and 'img' in args.input:
        x_train = [X_img_train, X_coord_train]
        x_validation = [X_img_validation, X_coord_validation]
        x_test = [X_img_test, X_coord_test]

        combined_model = concatenate([img_model.output, coord_model.output],name = 'cont_fusion_coord_img')
        z = Reshape((combined_model.shape[1], 1))(combined_model)
        z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu",name = 'conv1_fusion_coord_img')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = Flatten(name = 'flat_fusion_coord_img')(z)

        z = Dense(num_classes * 2, activation="relu", use_bias=True,name = 'dense1_fusion_coord_img')(z)
        z = Dropout(0.5,name = 'drop1_fusion_coord_lidar')(z)
        z = Dense(num_classes, activation="softmax", use_bias=True,name = 'dense2_fusion_coord_img')(z)
        model = Model(inputs=[img_model.input, coord_model.input], outputs=z)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        hist = model.fit(x_train, y_train,validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

        print(hist.history.keys())
        print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])

        print('***************Testing the model************')
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)

    else: # img+lidar
        x_train = [X_lidar_train,X_img_train]
        x_validation = [X_lidar_validation, X_img_validation]
        x_test = [X_lidar_test, X_img_test]

        combined_model = concatenate([lidar_model.output, img_model.output],name = 'cont_fusion_img_lidar')
        if args.fusion_architecture == 'cnn':
            z = Reshape((combined_model.shape[1], 1))(combined_model)
            a = z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu",name = 'conv_fusion_img_lidar')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
            z = Dropout(0.5,name = 'drop1_fusion_img_lidar')(z)
            z = Flatten(name = 'flat_fusion_img_lidar')(z)

            z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),name = 'dense1_fusion_img_lidar')(z)

        else:
            z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),name = 'dense2_fusion_img_lidar')(combined_model)  # USE THIS AND THE NEXT PART OF CODE OF mlp IMPLEMENTATION

        z = Dropout(0.5,name = 'drop2_fusion_img_lidar')(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_img_lidar')(z)

        model = Model(inputs=[lidar_model.input, img_model.input], outputs=z)
        model.compile(loss=categorical_crossentropy,optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()

        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.img_lidar'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

        print(hist.history.keys())
        print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])


        print('***************Testing the model************')
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)

elif multimodal == 3:
    x_train = [X_lidar_train,X_img_train,X_coord_train]
    x_validation = [X_lidar_validation, X_img_validation, X_coord_validation]
    x_test = [X_lidar_test, X_img_test, X_coord_test]

    combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output],name = 'cont_fusion_coord_img_lidar')
    if args.fusion_architecture == 'cnn':
        z = Reshape((combined_model.shape[1], 1))(combined_model)
        a= z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu",name = 'conv_fusion_coord_img_lidar')(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = Dropout(0.5,name = 'drop1_fusion_coord_img_lidar')(z)
        z = Flatten(name = 'flat_fusion_coord_img_lidar')(z)

        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),name = 'dense1_fusion_coord_img_lidar')(z)

    else:
        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),name = 'dense2_fusion_coord_img_lidar')(combined_model) # USE THIS AND THE NEXT PART OF CODE OF mlp IMPLEMENTATION

    z = Dropout(0.5,name = 'drop2_fusion_coord_img_lidar')(z)
    z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_img_lidar')(z)
    model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,top_50_accuracy,precision_m, recall_m, f1_m])
    model.summary()

    # TRAINING THE MODEL
    hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_lidar_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

    print(hist.history.keys())
    print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])

    print('***************Testing the model************')
    scores = model.evaluate(x_test, y_test)
    print(model.metrics_names, scores)
###############################################################################
# Single modalities
###############################################################################
else:

    if 'coord' in args.input:
        if args.strategy == 'reg':
            model = coord_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()

            hist = model.fit(X_coord_train,y_train,validation_data=(X_coord_validation, y_validation),
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses in train:', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_coord_test, y_test)
            pprint('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = coord_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
            model.summary()
            hist = model.fit(X_coord_train,y_train, validation_data=(X_coord_validation, y_validation),
            epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])

            print('***************Testing model************')
            scores = model.evaluate(X_coord_test, y_test)
            print(model.metrics_names,scores)
        # print('*****************Seperate statics***********************')
        # seperate_metric_in_out_train(model,X_coord_train,y_train,X_coord_validation, y_validation)


    elif 'img' in args.input:

        if args.strategy == 'reg':
            model = img_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()

            hist = model.fit(X_img_train,y_train, validation_data=(X_coord_validation, y_validation),
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses in train:', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_img_test, y_test)
            print('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = img_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
            model.summary()
            hist = model.fit(X_img_train,y_train, validation_data=(X_img_validation, y_validation),
            epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.img_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])

            print('***************Testing model************')
            scores = model.evaluate(X_img_test, y_test)
            print(model.metrics_names,scores)

        # print('*****************Seperate statics***********************')
        # seperate_metric_in_out_train(model,X_img_train,y_train,X_img_validation, y_validation)


    else: #LIDAR
        if args.strategy == 'reg':
            model = lidar_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()

            hist = model.fit(X_lidar_train,y_train,validation_data=(X_lidar_validation, y_validation),
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses in train:', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_lidar_test, y_test)
            print('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = lidar_model
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
            model.summary()
            hist = model.fit(X_lidar_train,y_train, validation_data=(X_lidar_validation, y_validation),epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle,callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.lidar.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto')])

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy']
                ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'])

            print('***************Testing model************')
            scores = model.evaluate(X_lidar_test, y_test)
            print(model.metrics_names,scores)

        # print('*****************Seperate statics***********************')
        # seperate_metric_in_out_train(model,X_lidar_train,y_train,X_lidar_validation, y_validation)

end = time()
print('The execusion time is:',end-start)

