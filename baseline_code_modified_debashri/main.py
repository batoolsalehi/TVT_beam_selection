from __future__ import division

import csv
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model, load_model
from tensorflow.keras.layers import Dense,concatenate, Dropout, Conv1D, Flatten, Reshape, Activation
from tensorflow.keras.losses import categorical_crossentropy
#from keras.utils.np_utils import to_categorical
from tensorflow.keras import regularizers
#from tensorflow.losses import mean_squared_error
from tensorflow.keras.optimizers import Adadelta,Adam, SGD, Nadam,Adamax, Adagrad
from tensorflow.python.keras.layers.normalization import BatchNormalization
import sklearn
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from ModelHandler import ModelHandler
import numpy as np
import argparse
from sklearn.model_selection import KFold
from tqdm import tqdm
import random
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import matplotlib
import h5py
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from custom_metrics import *
# from beam_selection.baseline_code_modified_debashri.utils.custom_metrics import *
############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
#tf.set_random_seed(seed)
#tf.random_set_seed()
np.random.seed(seed)
random.seed(seed)

from sklearn.feature_selection import SelectKBest

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

def top_2_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=2)

def top_10_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=10)

def top_50_accuracy(y_true,y_pred):
    return metrics.top_k_categorical_accuracy(y_true,y_pred,k=50)

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

    # yMatrix /= np.max(yMatrix)

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
            # y[i,:] = thisOutputs
            y[i,:] = logOut

    else:
        print('Invalid strategy')
    # scaler = RobustScaler()
    # scaler.fit(y)
    # y = scaler.transform(y)
    print('one sample', y[0,:])

    return y,num_classes


def balance_data(beams,modal,variance,dim):
    'This function balances the dataset by generating multiple copies of classes with low Apperance'

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


def meaure_topk_for_regression(y_true,y_pred,k):
    'Measure top 10 accuracy for regression'
    c = 0
    for i in range(len(y_pred)):
        # shape of each elemnt is (256,)
        A = y_true[i]
        B = y_pred[i]
        top_predictions = B.argsort()[-10:][::-1]
        best = np.argmax(A)
        if best in top_predictions:
             c +=1

    return c/len(y_pred)



parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str)
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=32, type=int,help='Batch size')

parser.add_argument('--epochs', default=100, type = int, help='Specify the epochs to train')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--Aug', type=str2bool, help='Do Augmentaion to balance the dataset or not', default=False)
parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--augmented_folder', help='Location of the augmeneted data', type=str, default='G:/Beam_Selection_ITU/beam_selection/baseline_code_modified/aug_data/')

parser.add_argument('--fusion_architecture', type=str ,default='mlp', help='Whether to use convolution in fusion network architecture',choices=['mlp','cnn'])
parser.add_argument('--k_fold', type=int, help='K-fold Cross validation', default=0)
parser.add_argument('--test_data_folder', help='Location of the test data directory', type=str)
parser.add_argument('--img_version', type=str, help='Which version of image folder to use', default='')
parser.add_argument('--loadWeights', type=str2bool, help='Load single modality trained weights', default=False)
parser.add_argument('--load_model_folder', help='Location of the trained models folder', type=str)

filepath = 'best_weights.wts.h5'









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
output_train_file = args.data_folder+'beam_output/beams_output_train.npz'
output_validation_file = args.data_folder+'beam_output/beams_output_validation.npz'
output_test_file = args.test_data_folder+'beam_output/beams_output_test.npz'


if args.strategy == 'one_hot':
    y_train,num_classes = custom_label(output_train_file,'one_hot')
    y_validation, _  = custom_label(output_validation_file,'one_hot')
    y_test, _ = custom_label(output_test_file, 'one_hot')
elif args.strategy == 'default':
    y_train,num_classes = getBeamOutput(output_train_file)
    y_validation, _ = getBeamOutput(output_validation_file)
elif args.strategy == 'reg':
    y_train,num_classes = custom_label(output_train_file,'reg')
    y_validation, _ = custom_label(output_validation_file,'reg')
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
    X_coord_test = open_npz(args.test_data_folder + 'coord_input/coord_test.npz', 'coordinates') # added for testing
    coord_train_input_shape = X_coord_train.shape

    if args.Aug:
        try:
            #train
            X_coord_train = open_npz(args.augmented_folder+'coord_input/coord_train.npz','train')
            y_train = open_npz(args.augmented_folder+'beam_output/beams_output_train.npz','train')

            #validation
            # X_coord_validation = open_npz(args.augmented_folder+'coord_input/coord_validation.npz','val')
            # y_validation = open_npz(args.augmented_folder+'beam_output/beams_output_validation.npz','val')
        except:
            print('****************Augment coordinates****************')
            y_train, X_coord_train = balance_data(Initial_labels_train,X_coord_train,0.001,(2,))
            # y_validation, X_coord_validation = balance_data(Initial_labels_val,X_coord_validation,0.001,(2,))
            save_npz(args.augmented_folder+'coord_input/','coord_train.npz',X_coord_train,'coord_validation.npz',X_coord_validation)
            # save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

    X_coord_train = X_coord_train.reshape((X_coord_train.shape[0], X_coord_train.shape[1], 1))
    X_coord_validation = X_coord_validation.reshape((X_coord_validation.shape[0], X_coord_validation.shape[1], 1))
    print(X_coord_train.shape)


if 'img' in args.input:
    ###############################################################################
    resizeFac = 20 # Resize Factor
    nCh = 1 # The number of channels of the image
    imgDim = (360,640) # Image dimensions

    image_folder = 'image_input'
    #train
    if args.img_version =='v2': image_folder = 'image_v2_input'
    X_img_train = open_npz(args.data_folder+image_folder +'/img_input_train_'+str(resizeFac)+'.npz','inputs')
    #validation
    X_img_validation = open_npz(args.data_folder+image_folder +'/img_input_validation_'+str(resizeFac)+'.npz','inputs')
    X_img_test = open_npz(args.test_data_folder + image_folder + '/img_input_test_' + str(resizeFac) + '.npz','inputs')
    img_train_input_shape = X_img_train.shape

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
            y_validation, X_img_validation = balance_data(Initial_labels_val,X_img_validation,0.001,(48, 81, 1))
            print('saving input')
            save_npz(args.augmented_folder+'image_input/','img_input_train_20.npz',X_img_train,'img_input_validation_20.npz',X_img_validation)
            print('saving Outputs')
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

if 'lidar' in args.input:
    ###############################################################################
    #train
    X_lidar_train = open_npz(args.data_folder+'lidar_input/lidar_train.npz','input')
    #validation
    X_lidar_validation = open_npz(args.data_folder+'lidar_input/lidar_validation.npz','input')
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
            y_validation, X_lidar_validation = balance_data(Initial_labels_val,X_lidar_validation,0.001,(20, 200, 10))
            save_npz(args.augmented_folder+'lidar_input/','lidar_train.npz',X_lidar_train,'lidar_validation.npz',X_lidar_validation)
            save_npz(args.augmented_folder+'beam_output/','beams_output_train.npz',y_train,'beams_output_validation.npz',y_validation)

##############################################################################
# Model configuration
##############################################################################
#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)
fusion = False if len(args.input) == 1 else True
# print(fusion)

validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt = Nadam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#opt = Adamax(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
#opt = Adagrad(lr=args.lr, epsilon=1e-07)
#opt = Adadelta(lr=args.lr, rho=0.95, epsilon=1e-07)
#opt= Adam(lr=args.lr)
file_name = 'acc'

if 'coord' in args.input:
    coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete',args.strategy, fusion)
    if args.loadWeights:
        coord_model.load_weights(args.load_model_folder + 'best_weights.coord.h5', by_name=True)
if 'img' in args.input:
    if nCh==1:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_train_input_shape[1],img_train_input_shape[2],1],'complete',args.strategy,fusion)
    else:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_train_input_shape[1],img_train_input_shape[2],img_train_input_shape[3]],'complete',args.strategy, fusion)
    if args.loadWeights:
        img_model.load_weights(args.load_model_folder + 'best_weights.img.h5', by_name=True)
if 'lidar' in args.input:
    lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy, fusion)
    # lidar_model.summary()
    # lidar_model.layers.pop()
    # lidar_model.layers = lidar_model.layers[:-1]
    # lidar_model.summary()
    # with h5py.File(args.load_model_folder + 'best_weights.lidar.h5.hdf5', "r") as f:
    #     print(f.keys())
    if args.loadWeights:
        lidar_model.load_weights(args.load_model_folder + 'best_weights.lidar.h5', by_name=True)


# ADDED TO PERFORM ONE HOT ENCODING
#y_train = to_categorical(y_train, num_classes=num_classes)
#y_validation = to_categorical(y_validation, num_classes=num_classes)





if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        file_name = "lidar_coord"
        combined_model = concatenate([lidar_model.output, coord_model.output])
        x_validation = [X_lidar_validation, X_coord_validation]
        x_train = [X_lidar_train, X_coord_train]
        x_test = [X_lidar_test, X_coord_test]

        # z = Reshape((combined_model.shape[1], 1))(combined_model)
        # z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu")(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        # z = Dropout(0.5)(z)
        # z = Flatten()(z)
        #
        # # z = Dropout(0.2)(z)
        # z = Dense(num_classes * 2, activation="relu", use_bias=True)(z)
        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(combined_model)
        z = Dropout(0.5)(z)
        z = Dense(num_classes, activation="softmax", use_bias=True)(z)
        model = Model(inputs=[lidar_model.input, coord_model.input], outputs=z)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy, top_2_accuracy,
                               metrics.top_k_categorical_accuracy, top_10_accuracy,
                               top_50_accuracy])
        model.summary()
        cb_list = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True,mode='auto'),
                   tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto',restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=2,mode='auto', min_lr=args.lr * 0.01)]

        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=cb_list)

        # PRINTING TEST RESULTS
        print('***************Validating the model************')
        scores = model.evaluate(x_validation, y_validation)
        print(model.metrics_names, scores)

    elif 'coord' in args.input and 'img' in args.input:
        file_name = "img_coord"
        combined_model = concatenate([img_model.output, coord_model.output])
        x_validation = [X_img_validation, X_coord_validation]
        x_test = [X_img_test, X_coord_test]
        z = Reshape((combined_model.shape[1], 1))(combined_model)
        z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu")(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        # z = Dropout(0.5)(z)
        z = Flatten()(z)

        # z = Dropout(0.2)(z)
        z = Dense(num_classes * 2, activation="relu", use_bias=True)(z)
        z = Dropout(0.5)(z)
        z = Dense(num_classes, activation="softmax", use_bias=True)(z)
        model = Model(inputs=[img_model.input, coord_model.input], outputs=z)
        model.compile(loss=categorical_crossentropy,
                      optimizer=opt,
                      metrics=[metrics.categorical_accuracy, top_2_accuracy,
                               metrics.top_k_categorical_accuracy, top_10_accuracy,
                               top_50_accuracy])
        model.summary()
        hist = model.fit([X_img_train, X_coord_train], y_train,
                         validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs,
                         callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_categorical_accuracy', verbose=2,save_best_only=True, mode='auto'),
                             tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=5, verbose=2,mode='auto')])


    # IMPLEMENTED K-FOLD CROSS VALIDATION
    else:
        file_name = 'lidar_img'
        x_test = [X_lidar_test, X_img_test]
        combined_model = concatenate([lidar_model.output, img_model.output])
        if args.fusion_architecture == 'cnn':
            z = Reshape((combined_model.shape[1], 1))(combined_model)
            a = z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu")(
                z)  # KERNEL SIZE CHANGED FROM 1 TO 2
            z = Dropout(0.5)(z)
            z = Flatten()(z)

            # z = Dropout(0.2)(z)
            z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

        else:
            z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(combined_model)  # USE THIS AND THE NEXT PART OF CODE OF mlp IMPLEMENTATION

        z = Dropout(0.5)(z)
        z = Dense(num_classes, activation="softmax")(z)

        model = Model(inputs=[lidar_model.input, img_model.input], outputs=z)
        model.compile(loss=categorical_crossentropy,optimizer=opt,
                      metrics=[metrics.categorical_accuracy, top_2_accuracy,
                               metrics.top_k_categorical_accuracy, top_10_accuracy, top_50_accuracy])
        model.summary()
        cb_list = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True,mode='auto'),
                   tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto',restore_best_weights=True),
                   tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=2,mode='auto', min_lr=args.lr * 0.01)]

        # CONCATINATING TRAIN AND VALIDATION AND PERFORM K FOLD CROSS VALIDATION
        k_folds = args.k_fold
        cvscores = []
        testscores = []
        # K fold cross Validation
        if args.k_fold >0:
            while k_folds >0:
                print("Starting ", k_folds, "th Fold........")
                x_lidar_data = np.concatenate([X_lidar_train, X_lidar_validation], axis=0)
                x_img_data = np.concatenate([X_img_train, X_img_validation], axis=0)
                y_data = np.concatenate([y_train, y_validation], axis=0)
                x_lidar_data, x_img_data, y_data = sklearn.utils.shuffle(x_lidar_data, x_img_data, y_data)
                trainLength = int(x_lidar_data.shape[0]*(1-validationFraction))
                print(trainLength)
                X_lidar_train, X_lidar_validation = x_lidar_data[0:trainLength, :], x_lidar_data[trainLength:x_lidar_data.shape[0], :]
                X_img_train, X_img_validation = x_img_data[0:trainLength, :], x_img_data[trainLength:x_img_data.shape[0], :]
                y_train, y_validation = y_data[0:trainLength, :], y_data[trainLength:y_data.shape[0], :]

                # TRAINING THE MODEL
                x_validation = [X_lidar_validation, X_img_validation]
                x_train = [X_lidar_train, X_img_train]
                hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=cb_list)

                # PRINTING TEST RESULTS
                print('***************Validating the model************')
                scores = model.evaluate(x_validation, y_validation)
                print(model.metrics_names, scores)
                # print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
                cvscores.append(scores[4] * 100)
                # K.clear_session()
                k_folds = k_folds - 1

                print('***************Testing the model************')
                scores = model.evaluate(x_test, y_test)
                print(model.metrics_names, scores)
                testscores.append(scores[4] * 100)

        else:
            # TRAINING THE MODEL
            x_validation = [X_lidar_validation, X_img_validation]
            x_train = [X_lidar_train,X_img_train]

            hist = model.fit(x_train,y_train,validation_data=(x_validation, y_validation), epochs=args.epochs,batch_size=args.bs, callbacks=cb_list)
            # PRINTING TEST RESULTS
            print('***************Validating the model************')
            scores = model.evaluate(x_validation, y_validation)
            print(model.metrics_names, scores)

            print('***************Testing the model************')
            scores = model.evaluate(x_test, y_test)
            print(model.metrics_names, scores)

elif multimodal == 3:

    file_name = "lidar_img_coord"
    combined_model = concatenate([lidar_model.output, img_model.output, coord_model.output])
    x_test = [X_lidar_test, X_img_test, X_coord_test]

    if args.fusion_architecture == 'cnn':
        z = Reshape((combined_model.shape[1], 1))(combined_model)
        a= z = Conv1D(num_classes * 2, kernel_size=1, strides=1, activation="relu")(z)  # KERNEL SIZE CHANGED FROM 1 TO 2
        z = Dropout(0.5)(z)
        z = Flatten()(z)

        # z = Dropout(0.2)(z)
        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(z)

    else:
        z = Dense(num_classes * 2, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(combined_model) # USE THIS AND THE NEXT PART OF CODE OF mlp IMPLEMENTATION

    z = Dropout(0.5)(z)
    z = Dense(num_classes, activation="softmax")(z)
    model = Model(inputs=[lidar_model.input, img_model.input, coord_model.input], outputs=z)
    model.compile(loss=categorical_crossentropy,
                  optimizer=opt,
                  metrics=[metrics.categorical_accuracy, top_2_accuracy,
                           metrics.top_k_categorical_accuracy, top_10_accuracy,
                           top_50_accuracy])
    model.summary()
    cb_list = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto',restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=2, mode='auto',min_lr=args.lr * 0.01)]

    # CONCATINATING TRAIN AND VALIDATION AND PERFORM K FOLD CROSS VALIDATION
    k_folds = args.k_fold
    cvscores = []
    testscores = []
    # K fold cross Validation
    if args.k_fold > 0:
        while k_folds > 0:
            print("Starting ", k_folds, "th Fold........")
            # CONCATINATING TRAIN AND VALIDATION AND PERFORM K FOLD CROSS VALIDATION
            x_lidar_data = np.concatenate([X_lidar_train, X_lidar_validation], axis=0)
            x_img_data = np.concatenate([X_img_train, X_img_validation], axis=0)
            x_coord_data = np.concatenate([X_coord_train, X_coord_validation], axis=0)
            y_data = np.concatenate([y_train, y_validation], axis=0)
            x_lidar_data, x_img_data, x_coord_data, y_data = sklearn.utils.shuffle(x_lidar_data, x_img_data, x_coord_data, y_data)
            trainLength = int(x_lidar_data.shape[0] * (1 - validationFraction))
            print(trainLength)
            X_lidar_train, X_lidar_validation = x_lidar_data[0:trainLength, :], x_lidar_data[trainLength:x_lidar_data.shape[0],:]
            X_img_train, X_img_validation = x_img_data[0:trainLength, :], x_img_data[trainLength:x_img_data.shape[0], :]
            X_coord_train, X_coord_validation = x_coord_data[0:trainLength, :], x_coord_data[trainLength:x_coord_data.shape[0], :]
            y_train, y_validation = y_data[0:trainLength, :], y_data[trainLength:y_data.shape[0], :]

            # TRAINING THE MODEL
            x_validation = [X_lidar_validation, X_img_validation, X_coord_validation]
            x_train = [X_lidar_train, X_img_train, X_coord_train]
            hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=cb_list)

            # PRINTING TEST RESULTS
            print('***************Validating the model************')
            scores = model.evaluate(x_validation, y_validation)
            print(model.metrics_names, scores)
            # print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
            cvscores.append(scores[4] * 100)
            #K.clear_session()
            k_folds = k_folds - 1

            print('***************Testing the model************')
            scores = model.evaluate(x_test, y_test)
            print(model.metrics_names, scores)
            # print("%s: %.2f%%" % (model.metrics_names[4], scores[4] * 100))
            testscores.append(scores[4] * 100)
    else:
        # TRAINING THE MODEL
        x_validation = [X_lidar_validation, X_img_validation, X_coord_validation]
        x_train = [X_lidar_train,X_img_train,X_coord_train]
        hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), epochs=args.epochs, batch_size=args.bs, callbacks=cb_list)

        # PRINTING TEST RESULTS
        print('***************Validating model************')
        scores = model.evaluate(x_validation, y_validation)
        print(model.metrics_names, scores)

        print('***************Testing the model************')
        scores = model.evaluate(x_test, y_test)
        print(model.metrics_names, scores)
else:
    if 'coord' in args.input:

        if args.strategy == 'reg':
            model = coord_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(X_coord_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_coord_validation, y_validation)
            print(model.metrics_names,scores)

            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_coord_validation)
            top_10_coord = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for coord is:',top_10_coord)


        if args.strategy == 'one_hot':
            model = coord_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy])
            model.summary()
            hist = model.fit(X_coord_train,y_train,
            epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle)

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'])
            print('top_2_accuracy',hist.history['top_2_accuracy'])
            print('top_10_accuracy', hist.history['top_10_accuracy'])

            print('***************Testing model************')
            scores = model.evaluate(X_coord_validation, y_validation)
            print(model.metrics_names,scores)


    elif 'img' in args.input:

        print('********************Normalize image********************')
        X_img_train = X_img_train.astype('float32') / 255
        X_img_validation = X_img_validation.astype('float32') / 255

        if args.strategy == 'reg':
            model = img_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(X_img_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_img_validation, y_validation)
            print(model.metrics_names,scores)


            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_img_validation)
            top_10_img = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for image is:',top_10_img)

        if args.strategy == 'one_hot':
            model = img_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy])
            model.summary()
            hist = model.fit(X_img_train,y_train,
            epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle)

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'])
            print('top_2_accuracy',hist.history['top_2_accuracy'])
            print('top_10_accuracy', hist.history['top_10_accuracy'])

            print('***************Testing model************')
            scores = model.evaluate(X_img_validation, y_validation)
            print(model.metrics_names,scores)



    else: #LIDAR
        file_name = "lidar"
        x_validation = X_lidar_validation
        x_train = X_lidar_train
        x_test = X_lidar_test
        if args.strategy == 'reg':
            model = lidar_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(x_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Validating***********************')
            scores = model.evaluate(x_validation, y_validation)
            print(model.metrics_names,scores)

            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_lidar_validation)
            top_10_lidar = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for lidar is:',top_10_lidar)


        if args.strategy == 'one_hot':
            model = lidar_model
            model.compile(loss=categorical_crossentropy,
                          optimizer=opt,
                          metrics=[metrics.categorical_accuracy, top_2_accuracy,
                                   metrics.top_k_categorical_accuracy, top_10_accuracy,
                                   top_50_accuracy])
            model.summary()
            cb_list = [tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='auto'),
                       tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=2, mode='auto',restore_best_weights=True),
                       tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=2,mode='auto', min_lr=args.lr * 0.01)]
            hist = model.fit(x_train, y_train, validation_data=(x_validation, y_validation),epochs=args.epochs,batch_size=args.bs, callbacks=cb_list, shuffle=args.shuffle)


            print('categorical_accuracy', hist.history['categorical_accuracy'],
                  'top_2_accuracy', hist.history['top_2_accuracy'],
                  'top_10_accuracy', hist.history['top_10_accuracy'])

            print('val_categorical_accuracy', hist.history['val_categorical_accuracy'],
                  'val_top_2_accuracy', hist.history['val_top_2_accuracy'],
                  'val_top_10_accuracy', hist.history['val_top_10_accuracy'])

            print('***************Validating model************')
            scores = model.evaluate(x_validation, y_validation)
            print(model.metrics_names, scores)

            print('***************Testing the model************')
            scores = model.evaluate(x_test, y_test)
            print(model.metrics_names, scores)

            # print('*****************Seperate statics***********************')
            # seperate_metric_in_out_train(model, X_lidar_train, y_train, X_lidar_validation, y_validation)


if args.k_fold > 0:
    print("Validation Scores")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print(cvscores)

    print("Test Scores")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(testscores), np.std(testscores)))
    print(testscores)

K.clear_session()

### SET PLOTTING PARAMETERS #########
params = {'legend.fontsize': 'small',
         'axes.labelsize': 'x-large',
         'axes.titlesize': 'x-large',
         'xtick.labelsize': 'x-large',
         'ytick.labelsize': 'x-large'}
plt.rcParams.update(params)


#Show Accuracy Curves
fig = plt.figure()
plt.title('Training Performance')
plt.plot(hist.epoch, hist.history['categorical_accuracy'], label='Top 1 Training Accuracy', linewidth=2.0, c='b')
plt.plot(hist.epoch, hist.history['val_categorical_accuracy'], label='Top 1 Validation Accuracy', linewidth=2.0, c='b',linestyle='--')
plt.plot(hist.epoch, hist.history['top_2_accuracy'], label='Top 2 Training Accuracy', linewidth=2.0, c='r')
plt.plot(hist.epoch, hist.history['val_top_2_accuracy'], label='Top 2 Validation Accuracy', linewidth=2.0, c='r',linestyle='--')
plt.plot(hist.epoch, hist.history['top_10_accuracy'], label='Top 10 Training Accuracy', linewidth=2.0, c='m')
plt.plot(hist.epoch, hist.history['val_top_10_accuracy'], label='Top 10 Validation Accuracy', linewidth=2.0, c='m',linestyle='--')
plt.plot(hist.epoch, hist.history['top_50_accuracy'], label='Top 50 Training Accuracy', linewidth=2.0, c='g')
plt.plot(hist.epoch, hist.history['val_top_50_accuracy'], label='Top 50 Validation Accuracy', linewidth=2.0, c='g',linestyle='--')

plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
fig.savefig(file_name+'_acc.png')  # save the figure to file
plt.close(fig)

# PRINT THE CONFUSION MATRIX
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.YlGnBu, labels=[], normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print("Normalized confusion matrix")
    else:
        cm = cm.astype('int')
    # print('Confusion matrix, without normalization')
    plt.rcParams.update(params) # ADDED
    fig = plt.figure(figsize=(20,20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    thresh = cm.max() / 2
    fmt = '.2f' if normalize else 'd'

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig.savefig(file_name+'_conf_mat.png')  # save the figure to file
    plt.close(fig)


# plt.show()
np.set_printoptions(threshold=np.inf)
classes = []
for i in range(num_classes):
	classes.append(str(i))

# Plot confusion matrix
#y_val_hat = np.around(model.predict(x_validation, batch_size=args.bs))
y_val_hat = model.predict(x_validation, batch_size=args.bs)
y_test_hat = model.predict(x_test, batch_size=args.bs)
#print(y_val_hat)

# PRINTING THE TRUE AND PREDICTED LABELS ACCURACY
# TRUE VALIDATION LEBELS
num_zero = 0
num_nonzero = 0
yval_dict = {}
yval_dict = dict.fromkeys(range(0, num_classes), 0)

for i in range(0, len(y_validation)):
    if 1 in list(y_validation[i, :]):
        j = list(y_validation[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_validation[i, :]))
    yval_dict[k] = yval_dict[k] + 1

print("Num of zeros in true labels", num_zero, " and non zeros ", num_nonzero)
print(yval_dict.values())
print(yval_dict.items())
fig = plt.figure()
plt.bar(yval_dict.keys(), yval_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_val.png')  # save the figure to file
plt.close(fig)

# PREDICTED VALIDATION LABELS
num_zero = 0
num_nonzero = 0
yval_hat_dict = {}
yval_hat_dict = dict.fromkeys(range(0, num_classes), 0)
#yval_dict.update()
#print(yval_hat_dict)
for i in range(0, len(y_val_hat)):
    if 1 in list(y_val_hat[i, :]):
        j = list(y_val_hat[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_val_hat[i, :]))
    yval_hat_dict[k] = yval_hat_dict[k] + 1

print("Num of zeros in predicted labels ", num_zero, " and non zeros ", num_nonzero)
print(yval_hat_dict.values())
print(yval_hat_dict.items())

fig = plt.figure()
plt.bar(yval_hat_dict.keys(), yval_hat_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_val_pred.png')  # save the figure to file
plt.close(fig)



# PREDICTED TEST LABELS
num_zero = 0
num_nonzero = 0
ytest_hat_dict = {}
ytest_hat_dict = dict.fromkeys(range(0, num_classes), 0)
#yval_dict.update()
#print(yval_hat_dict)
for i in range(0, len(y_test_hat)):
    if 1 in list(y_test_hat[i, :]):
        j = list(y_test_hat[i, :]).index(1)
        num_nonzero = num_nonzero + 1
    else:
        j = 0
        num_zero = num_zero + 1
    k = int(np.argmax(y_test_hat[i, :]))
    ytest_hat_dict[k] = ytest_hat_dict[k] + 1

print("Num of zeros in predicted labels ", num_zero, " and non zeros ", num_nonzero)
print(ytest_hat_dict.values())
print(ytest_hat_dict.items())

fig = plt.figure()
plt.bar(ytest_hat_dict.keys(), ytest_hat_dict.values(), color='green')
plt.xlabel("Labels")
plt.ylabel("Frequncy")
#plt.title("Energy output from various fuel sources")

#plt.xticks(x_pos, x)
plt.tight_layout()
fig.savefig(file_name+'_test_pred.png')  # save the figure to file
plt.close(fig)


conf = np.zeros([len(classes), len(classes)])
confnorm = np.zeros([len(classes), len(classes)])
for i in range(0, len(x_validation)):
    if 1 in list(y_val_hat[i, :]):
    	j = list(y_validation[i, :]).index(1)
    else:
	    j = 0
    k = int(np.argmax(y_val_hat[i, :]))
    conf[j, k] = conf[j, k] + 1
plot_confusion_matrix(conf, labels=classes, normalize=False)
