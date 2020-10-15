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

def add_model(model_flag, model,save_path):

    # save the model structure first
    model_json = model.to_json()
    print('\n*************** Saving New Model Structure ***************')
    with open(os.path.join(save_path, "%s_model.json" % model_flag), "w") as json_file:
        json_file.write(model_json)
        print("json file written")
        print(os.path.join(save_path, "%s_model.json" % model_flag))


# loading the model structure from json file
def load_model_structure(model_path='/scratch/model.json'):

    # reading model from json file
    json_file = open(model_path, 'r')
    model = model_from_json(json_file.read())
    json_file.close()
    return model


def load_weights(model, weight_path = '/scratch/weights.02-3.05.hdf5'):
    model.load_weights(weight_path)


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False

def open_npz(path,key):
    data = np.load(path)[key]
    return data

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


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--test_data_folder', help='Location of the test data directory', type=str)

parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)

parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--fusion_architecture', type=str ,default='mlp', help='Whether to use convolution in fusion network architecture',choices=['mlp','cnn'])
parser.add_argument('--img_version', type=str, help='Which version of image folder to use', default='')

parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = '/home/batool/beam_selection_NU/baseline_code/model_folder/')

parser.add_argument('--image_feature_to_use', type=str ,default='v1', help='feature images to use',choices=['v1','v2','custom'])
parser.add_argument('--only_predict', type=str2bool, help='only predict or evaluate', default=False)


args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)
###############################################################################
# Outputs
###############################################################################
# Read files
if not args.only_predict:
    output_test_file = args.test_data_folder+'beam_output/beams_output_test.npz'

    if args.strategy == 'default':
        y_test, _ = getBeamOutput(output_test_file)
    elif args.strategy == 'one_hot' or args.strategy == 'reg':
        y_test, _ = custom_label(output_test_file,args.strategy)

    else:
        print('invalid labeling strategy')

###############################################################################
# Inputs
###############################################################################

if 'coord' in args.input:
    #test
    X_coord_test = open_npz(args.test_data_folder + 'coord_input/coord_test.npz', 'coordinates')
    ### For convolutional input
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))
    print(X_coord_test.shape)


if 'img' in args.input:
    ###############################################################################
    resizeFac = 20 # Resize Factor
    if args.image_feature_to_use == 'v1':
        folder = 'image_input'
    elif args.image_feature_to_use == 'v2':
        folder =  'image_v2_input'
    elif args.image_feature_to_use == 'custom':
        folder = 'image_custom_input'

    #test
    X_img_test = open_npz(args.test_data_folder+folder+'/img_input_test_' + str(resizeFac) + '.npz','inputs')

    if args.image_feature_to_use == 'v1' or args.image_feature_to_use =='v2':
        print('********************Normalize image********************')
        X_img_test = X_img_test.astype('float32') / 255
    elif args.image_feature_to_use == 'custom':
        print('********************Reshape images for convolutional********************')
        X_img_test = X_img_test.reshape((X_img_test.shape[0], X_img_test.shape[1], X_img_test.shape[2],1))

if 'lidar' in args.input:
    ###############################################################################
    #test
    X_lidar_test = open_npz(args.test_data_folder + 'lidar_input/lidar_test.npz', 'input')
##############################################################################
# Model configuration
##############################################################################
#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)
fusion = False if len(args.input) == 1 else True

opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

###############################################################################
# Fusion
###############################################################################

if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
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

        coord_model = load_model_structure(args.model_folder+'coord_model.json')
        coord_model.load_weights(args.model_folder + 'best_weights.coord.h5', by_name=True)

        model = coord_model
        model.compile(loss=categorical_crossentropy,
                            optimizer=opt,
                            metrics=[metrics.categorical_accuracy,
                                    top_2_accuracy, top_10_accuracy,
                                    top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        if not args.only_predict:
            print('***************Evaluating model************')
            scores = model.evaluate(X_coord_test, y_test)
            print(model.metrics_names,scores)
        else:
            preds = model.predict(X_coord_test)
            np.save('coord_pred'+'.npy',preds)

    elif 'img' in args.input:

        img_model = load_model_structure(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json')
        img_model.load_weights(args.model_folder + 'best_weights.img_'+args.image_feature_to_use+'.h5', by_name=True)

        model = img_model
        model.compile(loss=categorical_crossentropy,
                            optimizer=opt,
                            metrics=[metrics.categorical_accuracy,
                                    top_2_accuracy, top_10_accuracy,
                                    top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        if not args.only_predict:
            print('***************Evaluating model************')
            scores = model.evaluate(X_img_test, y_test)
            print(model.metrics_names,scores)
        else:
            preds = model.predict(X_img_test)
            np.save('img_pred'+'.npy',preds)

    else: #LIDAR


        lidar_model = load_model_structure(args.model_folder+'lidar_model.json')
        lidar_model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)
        model = lidar_model
        model.compile(loss=categorical_crossentropy,
                        optimizer=opt,
                        metrics=[metrics.categorical_accuracy,
                                    top_2_accuracy, top_10_accuracy,
                                    top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        if not args.only_predict:
            print('***************Testing model************')
            scores = model.evaluate(X_lidar_test, y_test)
            print(model.metrics_names,scores)
        else:
            preds = model.predict(X_lidar_test)
            np.save('lidar_pred'+'.npy',preds)


