from __future__ import division

import csv
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.models import model_from_json,Model
from tensorflow.keras.layers import Dense,concatenate
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.losses import mean_squared_error
from tensorflow.keras.optimizers import Adadelta,Adam, SGD
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
############################
# Fix the seed
############################
seed = 0
os.environ['PYTHONHASHSEED']=str(seed)
tf.set_random_seed(seed)
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

parser.add_argument('--epochs', default=10, type = int, help='Specify the epochs to train')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)
parser.add_argument('--id_gpu', default=2, type=int, help='which gpu to use.')
parser.add_argument('--Aug', type=str2bool, help='Do Augmentaion to balance the dataset or not', default=False)
parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--augmented_folder', help='Location of the augmeneted data', type=str, default='/home/batool/beam_selection/baseline_code_modified/aug_data/')


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

if args.strategy == 'one_hot':
    y_train,num_classes = custom_label(output_train_file,'one_hot')
    y_validation, _  = custom_label(output_validation_file,'one_hot')
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


if 'img' in args.input:
    ###############################################################################
    resizeFac = 20 # Resize Factor
    nCh = 1 # The number of channels of the image
    imgDim = (360,640) # Image dimensions

    #train
    X_img_train = open_npz(args.data_folder+'image_input/img_input_train_'+str(resizeFac)+'.npz','inputs')
    #validation
    X_img_validation = open_npz(args.data_folder+'image_input/img_input_validation_'+str(resizeFac)+'.npz','inputs')
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

validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt= Adam(lr=args.lr)

if 'coord' in args.input:
    coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete',args.strategy)
if 'img' in args.input:
    if nCh==1:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_train_input_shape[1],img_train_input_shape[2],1],'complete',args.strategy)
    else:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_train_input_shape[1],img_train_input_shape[2],img_train_input_shape[3]],'complete',arg.strategy)
if 'lidar' in args.input:
    lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete',args.strategy)


if multimodal == 2:
    if 'coord' in args.input and 'lidar' in args.input:
        combined_model = concatenate([coord_model.output,lidar_model.output])
        z = Dense(num_classes,activation="relu")(combined_model)
        model = Model(inputs=[coord_model.input,lidar_model.input],outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy, top_10_accuracy,
                            top_50_accuracy])
        model.summary()
        hist = model.fit([X_coord_train,X_lidar_train],y_train,
        validation_data=([X_coord_validation, X_lidar_validation], y_validation),epochs=args.epochs,batch_size=args.bs)

    elif 'coord' in args.input and 'img' in args.input:
        combined_model = concatenate([coord_model.output,img_model.output])
        z = Dense(num_classes,activation="relu")(combined_model)
        model = Model(inputs=[coord_model.input,img_model.input],outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy, top_10_accuracy,
                            top_50_accuracy])
        model.summary()
        hist = model.fit([X_coord_train,X_img_train],y_train,
        validation_data=([X_coord_validation, X_img_validation], y_validation), epochs=args.epochs,batch_size=args.bs)


    else:
        combined_model = concatenate([lidar_model.output,img_model.output])
        z = Dense(num_classes,activation="relu")(combined_model)
        model = Model(inputs=[lidar_model.input,img_model.input],outputs=z)
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy, top_10_accuracy,
                            top_50_accuracy])
        model.summary()
        hist = model.fit([X_lidar_train,X_img_train],y_train,
        validation_data=([X_lidar_validation, X_img_validation], y_validation), epochs=args.epochs,batch_size=args.bs)
elif multimodal == 3:
    combined_model = concatenate([lidar_model.output,img_model.output, coord_model.output])
    z = Dense(num_classes,activation="relu")(combined_model)
    model = Model(inputs=[lidar_model.input,img_model.input, coord_model.input],outputs=z)
    model.compile(loss=categorical_crossentropy,
                optimizer=opt,
                metrics=[metrics.categorical_accuracy,
                        metrics.top_k_categorical_accuracy, top_10_accuracy,
                        top_50_accuracy])
    model.summary()
    hist = model.fit([X_lidar_train,X_img_train,X_coord_train],y_train,
            validation_data=([X_lidar_validation, X_img_validation, X_coord_validation], y_validation),
            epochs=args.epochs,batch_size=args.bs)

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



    else:

        if args.strategy == 'reg':
            model = lidar_model
            model.compile(loss="mse",optimizer=opt,metrics=[top_2_accuracy,top_10_accuracy,top_50_accuracy])
            model.summary()

            hist = model.fit(X_lidar_train,y_train,
            epochs=args.epochs, batch_size=args.bs, shuffle=args.shuffle)
            print('losses', hist.history['loss'])

            print('*****************Testing***********************')
            scores = model.evaluate(X_lidar_validation, y_validation)
            print(model.metrics_names,scores)

            print('*****************Manual measuring, its same as using top_2_accuracy***********************')
            preds = model.predict(X_lidar_validation)
            top_10_lidar = meaure_topk_for_regression(y_validation,preds,10)
            print('top 10 accuracy for lidar is:',top_10_lidar)


        if args.strategy == 'one_hot':
            model = lidar_model
            model.compile(loss=categorical_crossentropy,
                                optimizer=opt,
                                metrics=[metrics.categorical_accuracy,
                                        top_2_accuracy, top_10_accuracy,
                                        top_50_accuracy])
            model.summary()
            hist = model.fit(X_lidar_train,y_train,
            epochs=args.epochs,batch_size=args.bs, shuffle=args.shuffle)

            print(hist.history.keys())
            print('categorical_accuracy', hist.history['categorical_accuracy'])
            print('top_2_accuracy',hist.history['top_2_accuracy'])
            print('top_10_accuracy', hist.history['top_10_accuracy'])

            print('***************Testing model************')
            scores = model.evaluate(X_lidar_validation, y_validation)
            print(model.metrics_names,scores)

