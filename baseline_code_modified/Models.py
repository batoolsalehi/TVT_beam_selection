from __future__ import division

import six
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Conv2D, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers


def Baseline(input_shape, num_classes, strategy):
    dropProb=0.3
    input_lid = Input(shape = input_shape)
    layer = Conv2D(16,kernel_size=(7,7),
                   activation='relu',padding="SAME",input_shape=input_shape)(input_lid)
    layer = Conv2D(16, (5, 5), padding="SAME", activation='relu')(layer)
    layer = Conv2D(16, (5, 5), padding="SAME", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(32, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(32, (3, 3), padding="SAME", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(1, 2))(layer)
    layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu')(layer)
    
    layer = Flatten()(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(0.25)(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dropout(0.25)(layer)
    if strategy == 'one_hot':
        out = Dense(num_classes,activation='softmax')(layer)
    elif strategy == 'reg':
        out = Dense(num_classes)(layer)
    return Model(inputs = input_lid, outputs = out)    


def ResLike(input_shape, num_classes, strategy):
    dropProb=0.3
    channel=32 # 32 now is the best, better than 64, 16
    input_lid = Input(shape = input_shape)
    a = layer = Conv2D(channel,kernel_size=(3, 3),
                   activation='relu',padding="SAME",input_shape=input_shape)(input_lid)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + a
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    b = layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + b
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    c = layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + c
    layer = MaxPooling2D(pool_size=(1, 2))(layer)
    d = layer = Dropout(dropProb)(layer)

    # if add this layer, need 35 epochs to converge
    # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + d
    # layer = MaxPooling2D(pool_size=(1, 2))(layer)
    # e = layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + d
    
    layer = Flatten()(layer)
    layer = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer)
    layer = Dropout(0.2)(layer) # 0.25 is similar ... could try more values
    layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer)
    layer = Dropout(0.2)(layer) # 0.25 is similar ... could try more values
    if strategy == 'one_hot':
        out = Dense(num_classes,activation='softmax')(layer)
    elif strategy == 'reg':
        out = Dense(num_classes)(layer)
    return Model(inputs = input_lid, outputs = out)     