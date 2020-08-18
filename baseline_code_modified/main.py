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

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def beamsLogScale(y,thresholdBelowMax):
        # shape is (#,256)
        y_shape = y.shape

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

    print("Reading dataset...", output_file)
    output_cache_file = np.load(output_file)
    yMatrix = output_cache_file['output_classification']

    yMatrix = np.abs(yMatrix)
    yMatrix /= np.max(yMatrix)
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
    'This function balances the dataset by generating multiple copies of classes with low Apperance'

    Best_beam=[]    # a list of 9234 elements(0,1,...,256) with the index of best beam
    k = 1          # Augment on best beam
    for count,val in enumerate(beams):   # beams is 9234*256
        Index = val.argsort()[-k:][::-1]
        Best_beam.append(Index[0])

    Apperance = {i:Best_beam.count(i) for i in Best_beam}   #Selected beam:  count apperance of diffrent classes
    print(Apperance)
    Max_apperance = max(Apperance.values())                 # Class with highest apperance

    for i in tqdm(Apperance.keys()):
        ind = [ind for ind, value in enumerate(Best_beam) if value == i]    # Find elements which are equal to i
        randperm = np.random.RandomState(seed).permutation(int(Max_apperance-Apperance[i]))%len(ind)

        extension = (len(randperm),)+dim
        print('extension',extension)
        ADD_beam = np.empty((len(randperm), 256))

        ADD_modal = np.empty(extension)
        print('shapes',ADD_beam.shape, ADD_modal.shape)
        for couter,v in enumerate(randperm):
            ADD_beam[couter,:] = beams[ind[v]]
            ADD_modal[couter,:] = modal[ind[v]]+variance*np.random.rand(*dim)
        beams = np.concatenate((beams, ADD_beam), axis=0)
        modal = np.concatenate((modal, ADD_modal), axis=0)

    randperm = np.random.RandomState(seed).permutation(len(beams))
    beams = beams[randperm]
    modal = modal[randperm]

    # print('Check class diversity After augmentation')
    # Best_beam_augmented=[]    # a list of 9234 elements with the index of best beam
    # k = 1          # Augment on best beam
    # for count,val in enumerate(beams):   # beams is 9234*256
    #     Index = val.argsort()[-k:][::-1]
    #     Best_beam_augmented.append(Index[0])

    # Apperance = {i:Best_beam_augmented.count(i) for i in Best_beam_augmented}   # Apprenace count
    # print(Apperance)

    return beams, modal


parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--data_folder', help='Location of the data directory', type=str)
#TODO: limit the number of input to 3
parser.add_argument('--input', nargs='*', default=['coord'],choices = ['img', 'coord', 'lidar'],
help='Which data to use as input. Select from: img, lidar or coord.')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--epochs', default=50, type = int, help='Specify the epochs to train')
parser.add_argument('--shuffle', help='shuffle or not', type=str2bool, default =True)
parser.add_argument('--id_gpu', default=2, type=int, help='which gpu to use.')
parser.add_argument('--Aug', type=str2bool, help='Do Augmentaion to balance the dataset or not', default=False)
parser.add_argument('--strategy', type=str ,default='one_hot', help='labeling strategy to use',choices=['baseline','one_hot','reg'])
parser.add_argument('--restore_aug_data', type=str2bool, help='restore augmented data or not', default=False)
parser.add_argument('--augmented_folder', help='Location of the augmeneted data', type=str, default='/home/batool/beam_selection/baseline_code_modified/aug_data/')


args = parser.parse_args()
print('Argumen parser inputs', args)

if args.id_gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

data_dir = args.data_folder


if 'coord' in args.input:
    ###############################################################################
    # Coordinate configuration
    #train
    coord_train_input_file = data_dir+'coord_input/coord_train.npz'
    coord_train_cache_file = np.load(coord_train_input_file)
    X_coord_train = coord_train_cache_file['coordinates']
    #validation
    coord_validation_input_file = data_dir+'coord_input/coord_validation.npz'
    coord_validation_cache_file = np.load(coord_validation_input_file)
    X_coord_validation = coord_validation_cache_file['coordinates']

    coord_train_input_shape = X_coord_train.shape

if 'img' in args.input:
    ###############################################################################
    # Image configuration
    resizeFac = 20 # Resize Factor
    nCh = 1 # The number of channels of the image
    imgDim = (360,640) # Image dimensions
    method = 1
    #train
    img_train_input_file = data_dir+'image_input/img_input_train_'+str(resizeFac)+'.npz'
    print("Reading dataset... ",img_train_input_file)
    img_train_cache_file = np.load(img_train_input_file)
    X_img_train = img_train_cache_file['inputs']
    #validation
    img_validation_input_file = data_dir+'image_input/img_input_validation_'+str(resizeFac)+'.npz'
    print("Reading dataset... ",img_validation_input_file)
    img_validation_cache_file = np.load(img_validation_input_file)
    X_img_validation = img_validation_cache_file['inputs']

    img_train_input_shape = X_img_train.shape


if 'lidar' in args.input:
    ###############################################################################
    # LIDAR configuration
    #train
    lidar_train_input_file = data_dir+'lidar_input/lidar_train.npz'
    print("Reading dataset... ",lidar_train_input_file)
    lidar_train_cache_file = np.load(lidar_train_input_file)
    X_lidar_train = lidar_train_cache_file['input']
    #validation
    lidar_validation_input_file = data_dir+'lidar_input/lidar_validation.npz'
    print("Reading dataset... ",lidar_validation_input_file)
    lidar_validation_cache_file = np.load(lidar_validation_input_file)
    X_lidar_validation = lidar_validation_cache_file['input']

    lidar_train_input_shape = X_lidar_train.shape


###############################################################################
# Output configuration
#train
output_train_file = data_dir+'beam_output/beams_output_train.npz'
output_validation_file = data_dir+'beam_output/beams_output_validation.npz'

if args.custom_label:
    y_train,num_classes = custom_label(output_train_file)
    y_validation, _  = custom_label(output_validation_file)
else:
    y_train,num_classes = getBeamOutput(output_train_file)
    y_validation, _ = getBeamOutput(output_validation_file)


##############################################################################
# Model configuration
##############################################################################

#multimodal
multimodal = False if len(args.input) == 1 else len(args.input)

num_epochs = args.epochs
batch_size = 32
validationFraction = 0.2 #from 0 to 1
modelHand = ModelHandler()
opt = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#opt= Adam(lr=args.lr)

if 'coord' in args.input:
    coord_model = modelHand.createArchitecture('coord_mlp',num_classes,coord_train_input_shape[1],'complete')
if 'img' in args.input:
   #num_epochs = 5
    if nCh==1:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_train_input_shape[1],img_train_input_shape[2],1],'complete')
    else:
        img_model = modelHand.createArchitecture('light_image',num_classes,[img_train_input_shape[1],img_train_input_shape[2],img_train_input_shape[3]],'complete')
if 'lidar' in args.input:
    lidar_model = modelHand.createArchitecture('lidar_marcus',num_classes,[lidar_train_input_shape[1],lidar_train_input_shape[2],lidar_train_input_shape[3]],'complete')


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
        validation_data=([X_coord_validation, X_lidar_validation], y_validation),epochs=num_epochs,batch_size=batch_size)

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
        validation_data=([X_coord_validation, X_img_validation], y_validation), epochs=num_epochs,batch_size=batch_size)


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
        validation_data=([X_lidar_validation, X_img_validation], y_validation), epochs=num_epochs,batch_size=batch_size)
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
            epochs=num_epochs,batch_size=batch_size)

else:
    if 'coord' in args.input:
        model = coord_model
        model.compile(loss=categorical_crossentropy,
                            optimizer=opt,
                            metrics=[metrics.categorical_accuracy,
                                    top_2_accuracy, top_10_accuracy,
                                    top_50_accuracy])
        model.summary()
        b,c = balance_data(y_train,X_coord_train)
        X_coord_train = c
        y_train = b

        randperm = np.random.permutation(len(X_coord_train))
        y_train = y_train[randperm]
        X_coord_train = X_coord_train[randperm]

        hist = model.fit(X_coord_train,y_train,
        #validation_data=(X_coord_validation, y_validation),
        epochs=num_epochs,batch_size=batch_size, shuffle=args.shuffle)
        prediction = model.predict(X_coord_train)


        # for i in prediction:
        #     print(np.where(i==max(i)))


    elif 'img' in args.input:
        model = img_model
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy, top_10_accuracy,
                            top_50_accuracy])
        model.summary()
        hist = model.fit(X_img_train,y_train,
        validation_data=(X_img_validation, y_validation),epochs=num_epochs,batch_size=batch_size)
        prediction = model.predict(X_img_validation)
        for i in prediction:
            max_index = i.argsort()[-1:][::-1]
            print(max_index)

    else:
        model = lidar_model
        model.compile(loss=categorical_crossentropy,
                    optimizer=opt,
                    metrics=[metrics.categorical_accuracy,
                            metrics.top_k_categorical_accuracy, top_10_accuracy,
                            top_50_accuracy])
        model.summary()
        hist = model.fit(X_lidar_train,y_train,
        validation_data=(X_lidar_validation, y_validation),epochs=num_epochs,batch_size=batch_size)





