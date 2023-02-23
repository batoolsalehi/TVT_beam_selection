########################################################
#Project name: Deep learning on multimodal sensor data at the wireless edge for vehicular network (IEEE Transactions on Vehicular Technology 2022)
# Contact: bsalehihikouei@ece.neu.edu
########################################################
from __future__ import division

import os
import csv
import argparse
import h5py
import pickle
import numpy as np
from tqdm import tqdm
import random
from time import time
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model, load_model
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation,multiply,MaxPooling1D,Add,AveragePooling1D,Lambda,Permute
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.initializers import glorot_uniform

import sklearn
from tensorflow.keras import backend as K
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import normalize

from ModelHandler import add_model,load_model_structure, ModelHandler
from custom_metrics import *
from math import sqrt, log
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as Marmar
from tensorflow.keras import backend as K
tf.compat.v1.disable_eager_execution()  #TF2

############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
tf.random.set_seed(seed)
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
    y_non_on_hot =  np.array(y)
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
    return y_non_on_hot,y,num_classes


def over_k(true,pred):  # for TF2
    ####compute accuracy per K
    dicti = {}
    for kth in range(256):
        kth_accuracy = metrics.top_k_categorical_accuracy(true,pred,k=kth)
        with tf.compat.v1.Session() as sess: this = kth_accuracy.eval()
        dicti[kth] =sum(this)/len(this)
    return dicti

def througput_ratio(preds, y):
    ####compute throughput ratio
    throughputs = {}
    for k in tqdm(range(1,256)):
        up = []
        down = []
        for exp in range(len(y)):
            true_1= y[exp].argsort()[-1:][::-1]
            t1 = log(y[exp,true_1]+1,2)

            top_preds = preds[exp].argsort()[-k:][::-1]
            p1 = max([log(y[exp,t]+1,2) for t in top_preds])
            up.append(p1)
            down.append(t1)

        throughputs['choices_'+str(k)] = sum(up)/sum(down)
    return(throughputs)


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--test_data_folder', help='Location of the test data directory', type=str)
parser.add_argument('--restore_models', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--epochs', default=10, type = int, help='Specify the epochs to train')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)
parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--img_version', type=str, help='Which version of image folder to use', default='')
parser.add_argument('--model_folder', help='Location of the trained models folder', type=str,default = '/home/batool/beam_selection_NU/baseline_code/model_folder/')
parser.add_argument('--image_feature_to_use', type=str ,default='v1', help='feature images to use',choices=['v1','v2','custom'])
parser.add_argument('--train_or_test', type=str ,default='test', help='Train or test',choices=['train','test'])

args = parser.parse_args()
print('Argumen parser inputs', args)


if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

check_and_create(args.model_folder)
###############################################################################
# Outputs (Beams)
###############################################################################
output_train_file = args.data_folder+'beam_output/beams_output_train.npz'
output_validation_file = args.data_folder+'beam_output/beams_output_validation.npz'
output_test_file = args.test_data_folder+'beam_output/beams_output_test.npz'
if args.strategy == 'default':
    y_train,num_classes = getBeamOutput(output_train_file)
    y_validation, _ = getBeamOutput(output_validation_file)
    y_test, _ = getBeamOutput(output_test_file)
elif args.strategy == 'one_hot' or args.strategy == 'reg':
    y_train_not_onehot,y_train,num_classes = custom_label(output_train_file,args.strategy)
    y_validation_not_onehot,y_validation, _ = custom_label(output_validation_file,args.strategy)
    y_test_not_onehot,y_test, _ = custom_label(output_test_file,args.strategy)

else:
    print('invalid labeling strategy')
###############################################################################
# Inputs (GPS, Image, LiDAR)
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
    ###############Normalize
    X_coord_train = normalize(X_coord_train, axis=1, norm='l1')
    X_coord_validation = normalize(X_coord_validation, axis=1, norm='l1')
    X_coord_test = normalize(X_coord_test, axis=1, norm='l1')
    ## Reshape for convolutional input
    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
    X_coord_test = X_coord_test.reshape((X_coord_test.shape[0], X_coord_test.shape[1], 1))

if 'img' in args.input:
    resizeFac = 20 # Resize Factor
    if args.image_feature_to_use == 'v1':
        folder = 'image_input'
    elif args.image_feature_to_use == 'v2':
        folder =  'image_v2_input'
    elif args.image_feature_to_use == 'custom':
        folder = 'image_custom_input'
    #train
    X_img_train = open_npz(args.data_folder+folder +'/img_input_train_'+str(resizeFac)+'.npz','inputs')/3
    #validation
    X_img_validation = open_npz(args.data_folder+folder+'/img_input_validation_'+str(resizeFac)+'.npz','inputs')/3
    #test
    X_img_test = open_npz(args.test_data_folder+folder+'/img_input_test_' + str(resizeFac) + '.npz','inputs')/3
    img_train_input_shape = X_img_train.shape

    if args.image_feature_to_use == 'v1' or args.image_feature_to_use =='v2':
        print('********************Normalize image********************')
        X_img_train = X_img_train.astype('float32') / 255
        X_img_validation = X_img_validation.astype('float32') / 255
        X_img_test = X_img_test.astype('float32') / 255
    elif args.image_feature_to_use == 'custom':
        print('********************Reshape images for convolutional********************')
        X_img_train = X_img_train.reshape((X_img_train.shape[0], X_img_train.shape[1], X_img_train.shape[2],1))/3
        X_img_validation = X_img_validation.reshape((X_img_validation.shape[0], X_img_validation.shape[1],X_img_validation.shape[2], 1))/3
        X_img_test = X_img_test.reshape((X_img_test.shape[0], X_img_test.shape[1], X_img_test.shape[2],1))/3

if 'lidar' in args.input:
    #train
    X_lidar_train = open_npz(args.data_folder+'lidar_input/lidar_train.npz','input')/2
    #validation
    X_lidar_validation = open_npz(args.data_folder+'lidar_input/lidar_validation.npz','input')/2
    #test
    X_lidar_test = open_npz(args.test_data_folder + 'lidar_input/lidar_test.npz', 'input')/2
    lidar_train_input_shape = X_lidar_train.shape

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
        if not os.path.exists(args.model_folder+'coord_model.json'):
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
        if not os.path.exists(args.model_folder+'image_'+args.image_feature_to_use+'_model'+'.json'):
            add_model('image_'+args.image_feature_to_use,img_model,args.model_folder)

if 'lidar' in args.input:
    if args.restore_models:
        lidar_model = load_model_structure(args.model_folder+'lidar_model.json')
        lidar_model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)
    else:
        lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy, fusion)
        if not os.path.exists(args.model_folder+'lidar_model.json'):
            add_model('lidar',lidar_model,args.model_folder)

###############################################################################
# Fusion Models
###############################################################################
if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        x_train = [X_lidar_train, X_coord_train]
        x_validation = [X_lidar_validation, X_coord_validation]
        x_test = [X_lidar_test, X_coord_test]

        combined_model = concatenate([lidar_model.output, coord_model.output],name = 'cont_fusion_coord_lidar')
        z = Reshape((2, 256))(combined_model)
        z = Permute((2, 1), input_shape=(2, 256))(z)
        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv1_fusion_coord_lid')(z)
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv2_fusion_coord_lid')(z)
        z = BatchNormalization()(z)
        z = MaxPooling1D(name='fusion_coord_lid_maxpool1')(z)

        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv3_fusion_coord_lid')(z)
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv4_fusion_coord_lid')(z)
        z = MaxPooling1D(name='fusion_coord_lid_maxpool2')(z)

        z = Flatten(name = 'flat_fusion_coord_lid')(z)
        z = Dense(num_classes * 3, activation="relu", use_bias=True,name = 'dense1_fusion_coord_lid')(z)
        z = Dropout(0.25,name = 'drop1_fusion_coord_lid')(z)
        z = Dense(num_classes * 2, activation="relu",name = 'dense2_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)
        z = Dropout(0.25,name = 'drop2_fusion_coord_img')(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        model = Model(inputs=[lidar_model.input, coord_model.input], outputs=z)
        add_model('coord_lidar',model,args.model_folder)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy, top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy])
        model.summary()
        if args.train_or_test=='train':
            print('***************Training************')
            hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_lidar.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                    ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
        elif args.train_or_test=='test':
            print('***************Testing************')
            model.load_weights(args.model_folder+'best_weights.coord_lidar.h5', by_name=True)
            scores = model.evaluate(x_test, y_test)
            print("Test results:", model.metrics_names, scores)


    elif 'coord' in args.input and 'img' in args.input:
        x_train = [X_img_train, X_coord_train]
        x_validation = [X_img_validation, X_coord_validation]
        x_test = [X_img_test, X_coord_test]
        combined_model = concatenate([img_model.output, coord_model.output],name = 'cont_fusion_coord_img')
        z = Reshape((2, 256))(combined_model)
        z = Permute((2, 1), input_shape=(2, 256))(z)
        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv1_fusion_coord_img')(z)
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv2_fusion_coord_img')(z)
        z = MaxPooling1D(name='fusion_coord_img_maxpool1')(z)

        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv3_fusion_coord_img')(z)
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv4_fusion_coord_img')(z)
        z = MaxPooling1D(name='fusion_coord_img_maxpool2')(z)

        z = Flatten(name = 'flat_fusion_coord_img')(z)
        z = Dense(num_classes * 3, activation="relu", use_bias=True,name = 'dense1_fusion_coord_img')(z)
        z = Dropout(0.25,name = 'drop1_fusion_coord_lidar')(z)
        z = Dense(num_classes * 2, activation="relu",name = 'dense2_fusion_coord_img', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)
        z = Dropout(0.25,name = 'drop2_fusion_coord_img')(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_img', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        model = Model(inputs=[img_model.input, coord_model.input], outputs=z)
        add_model('coord_img_custom',model,args.model_folder)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        if args.train_or_test=='train':
            print('***************Training************')
            hist = model.fit(x_train, y_train,validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2, mode='auto')])
            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                    ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
        elif args.train_or_test=='test':
            print('***************Testing************')
            model.load_weights(args.model_folder+'best_weights.coord_img_'+args.image_feature_to_use+'.h5', by_name=True)
            scores = model.evaluate(x_test, y_test)
            print("Test results:",model.metrics_names, scores)


    else: # img+lidar
        x_train = [X_lidar_train,X_img_train]
        x_validation = [X_lidar_validation, X_img_validation]
        x_test = [X_lidar_test, X_img_test]

        combined_model = concatenate([lidar_model.output, img_model.output],name = 'cont_fusion_img_lidar')
        z = Reshape((2, 256))(combined_model)
        z = Permute((2, 1), input_shape=(2, 256))(z)
        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv1_fusion_img_lid')(z)
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv2_fusion_img_lid')(z)
        z = BatchNormalization()(z)
        z = MaxPooling1D(name='fusion_img_lid_maxpool1')(z)

        z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv3_fusion_img_lid')(z)
        z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv4_fusion_img_lid')(z)
        z = MaxPooling1D(name='fusion_img_lid_maxpool2')(z)

        z = Flatten(name = 'flat_fusion_img_lid')(z)
        z = Dense(num_classes * 3, activation="relu", use_bias=True,name = 'dense1_fusion_img_lid')(z)
        z = Dropout(0.25,name = 'drop1_fusion_img_lidr')(z)
        z = Dense(num_classes * 2, activation="relu",name = 'dense2_fusion_img_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)
        z = Dropout(0.25,name = 'drop2_fusion_img_lid')(z)
        z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_img_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        model = Model(inputs=[lidar_model.input, img_model.input], outputs=z)
        add_model('img_lidar',model,args.model_folder)
        model.compile(loss=categorical_crossentropy,optimizer=opt,
                      metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy ,top_10_accuracy,top_25_accuracy,top_50_accuracy,precision_m, recall_m, f1_m])
        model.summary()
        if args.train_or_test=='train':
            print('***************Training************')
            hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.img_lidar'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                    ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])

        elif args.train_or_test=='test':
            print('***************Testing the model************')
            model.load_weights(args.model_folder+'best_weights.img_lidar'+args.image_feature_to_use+'.h5', by_name=True)
            scores = model.evaluate(x_test, y_test)
            print("Test results",model.metrics_names, scores)


elif multimodal == 3:
    x_train = [X_lidar_train,X_img_train,X_coord_train*0]
    x_validation = [X_lidar_validation, X_img_validation, X_coord_validation*0]
    x_test = [X_lidar_test, X_img_test, X_coord_test*0]
    combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output])
    z =check_shape= Reshape((3, 256))(combined_model)
    z = Permute((2, 1), input_shape=(3, 256))(z)
    z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv1_fusion_coord_lid')(z)
    z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv2_fusion_coord_lid')(z)
    z = BatchNormalization()(z)
    z = MaxPooling1D(name='fusion_coord_lid_maxpool1')(z)

    z = Conv1D(30, kernel_size=7, strides=1, activation="relu",name = 'conv3_fusion_coord_lid')(z)
    z = Conv1D(30, kernel_size=5, strides=1, activation="relu",name = 'conv4_fusion_coord_lid')(z)
    z = MaxPooling1D(name='fusion_coord_lid_maxpool2')(z)

    z = Flatten(name = 'flat_fusion_coord_lid')(z)
    z = Dense(num_classes * 3, activation="relu", use_bias=True,name = 'dense1_fusion_coord_lid')(z)
    z = Dropout(0.25,name = 'drop1_fusion_coord_lid')(z)            # # z = Dense(num_classes, activation="softmax", use_bias=True,name = 'dense2_fusion_coord_img')(z)
    z = Dense(num_classes * 2, activation="relu",name = 'dense2_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)
    z = Dropout(0.25,name = 'drop2_fusion_coord_img')(z)
    z = Dense(num_classes, activation="softmax",name = 'dense3_fusion_coord_lid', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

    model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
    add_model('coord_img_custom_lidar',model,args.model_folder)
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy,top_25_accuracy,top_50_accuracy,precision_m, recall_m, f1_m])
    model.summary()

    if args.train_or_test=='train':
        print('***************Training************')
        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord_img_lidar_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, verbose=2,mode='auto')])

        print(hist.history.keys())
        print('loss',hist.history['loss'],'val_loss',hist.history['val_loss'],'categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                    ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
    elif args.train_or_test=='test':
        print('***************Testing the model************')
        model.load_weights(args.model_folder+'best_weights.coord_img_lidar_'+args.image_feature_to_use+'.h5', by_name=True)
        scores = model.evaluate(x_test, y_test)
        print("Test results",model.metrics_names, scores)

###############################################################################
# Single modalities
###############################################################################
else:

    if 'coord' in args.input:
        if args.strategy == 'reg':
            model = coord_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()
            if args.train_or_test=='train':
                print('***************Training************')
                hist = model.fit(X_coord_train,y_train,validation_data=(X_coord_validation, y_validation),
                epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
                print('losses in train:', hist.history['loss'])
            elif args.train_or_test=='test':
                print('*****************Testing***********************')
                scores = model.evaluate(X_coord_test, y_test)
                pprint('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = coord_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
            model.summary()

            call_backs = []
            if args.train_or_test=='train':
                print('***************Training************')
                hist = model.fit(X_coord_train,y_train, validation_data=(X_coord_validation, y_validation),
                epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.coord.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto')])

                print(hist.history.keys())
                print['val_loss',hist.history['val_loss']]
                print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                        ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
            elif args.train_or_test=='test':
                print('***************Testing************')
                model.load_weights(args.model_folder + 'best_weights.coord.h5', by_name=True)   ## Restoring best weight for testing
                scores = model.evaluate(X_coord_test, y_test)
                print("Test results",model.metrics_names,scores)

    elif 'img' in args.input:

        if args.strategy == 'reg':
            model = img_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()
            if args.train_or_test=='train':
                print('***************Training************')
                hist = model.fit(X_img_train,y_train, validation_data=(X_coord_validation, y_validation),
                epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
                print('losses in train:', hist.history['loss'])
            elif args.train_or_test=='test':
                print('*****************Testing***********************')
                scores = model.evaluate(X_img_test, y_test)
                print('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            model = img_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy, top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
            model.summary()
            if args.train_or_test=='train':
                print('***************Training************')
                hist = model.fit(X_img_train,y_train, validation_data=(X_img_validation, y_validation),
                epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle, callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.img_'+args.image_feature_to_use+'.h5', monitor='val_loss', verbose=1, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2, mode='auto')])


                print(hist.history.keys())
                print('categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                        ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
            elif args.train_or_test=='test':
                print('*****************Testing***********************')
                model.load_weights(args.model_folder + 'best_weights.img_'+args.image_feature_to_use+'.h5', by_name=True)
                scores = model.evaluate(X_img_test, y_test)
                print("Test results",model.metrics_names,scores)

    else: #LIDAR
        if args.strategy == 'reg':
            model = lidar_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_1_accuracy,top_2_accuracy,top_10_accuracy,top_50_accuracy,R2_metric])
            model.summary()
            if args.train_or_test=='train':
                print('***************Training************')
                hist = model.fit(X_lidar_train,y_train,validation_data=(X_lidar_validation, y_validation),
                epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
                print('losses in train:', hist.history['loss'])
            elif args.train_or_test=='test':
                print('*****************Testing***********************')
                scores = model.evaluate(X_lidar_test, y_test)
                print('scores while testing:', model.metrics_names,scores)


        if args.strategy == 'one_hot':
            print('All shapes',X_lidar_train.shape,y_train.shape,X_lidar_validation.shape,y_validation.shape,X_lidar_test.shape,y_test.shape)
            model = lidar_model
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_5_accuracy,top_10_accuracy, top_25_accuracy,
                                        top_50_accuracy,precision_m, recall_m, f1_m])
            model.summary()
            if args.train_or_test=='train':
                print('***************Training************')
                hist = model.fit(X_lidar_train,y_train, validation_data=(X_lidar_validation, y_validation),epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle,callbacks=[tf.keras.callbacks.ModelCheckpoint(args.model_folder+'best_weights.lidar.h5', monitor='val_loss', verbose=2, save_best_only=True,mode='auto'),tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=2,mode='auto')])

                print(hist.history.keys())
                print('loss',hist.history['loss'],'val_loss',hist.history['val_loss'],'categorical_accuracy', hist.history['categorical_accuracy'],'top_2_accuracy',hist.history['top_2_accuracy'],'top_5_accuracy',hist.history['top_5_accuracy'],'top_10_accuracy', hist.history['top_10_accuracy'],'top_25_accuracy',hist.history['top_25_accuracy'],'top_50_accuracy',hist.history['top_50_accuracy']
                        ,'val_categorical_accuracy',hist.history['val_categorical_accuracy'],'val_top_2_accuracy',hist.history['val_top_2_accuracy'],'val_top_5_accuracy',hist.history['val_top_5_accuracy'],'val_top_10_accuracy',hist.history['val_top_10_accuracy'],'val_top_25_accuracy',hist.history['val_top_25_accuracy'],'val_top_50_accuracy',hist.history['val_top_50_accuracy'])
            elif args.train_or_test=='test':
                print('***************Testing************')
                model.load_weights(args.model_folder + 'best_weights.lidar.h5', by_name=True)   # to be added
                scores = model.evaluate(X_lidar_test, y_test)
                print("Test results",model.metrics_names,scores)

