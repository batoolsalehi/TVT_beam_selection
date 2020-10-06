from __future__ import division

import six
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Conv2D, Add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential

from tensorflow.keras import regularizers

#from tensorflow.keras.layers.convolutional import (
#    Conv2D,
#   MaxPooling2D,
#    AveragePooling2D
#)
#from keras.layers.merge import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

# DOUBLED THE NUMBER WITH NEURONS
def Baseline(input_shape, num_classes, strategy, fusion=False):
    dropProb=0.5
    channel = 16
    input_lid = Input(shape = input_shape)
    layer = Conv2D(channel,kernel_size=(7,7),activation='relu',padding="SAME",input_shape=input_shape)(input_lid)
    layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (5, 5), padding="SAME", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(2*channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(2*channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(4*channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(4*channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = MaxPooling2D(pool_size=(1, 2))(layer)
    layer = Dropout(dropProb)(layer)
    
    layer = Conv2D(4*channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(4*channel, (3, 3), padding="SAME", activation='relu')(layer)
    
    layer = Flatten()(layer)
    layer = Dense(num_classes*4, activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(num_classes*2, activation='relu')(layer) #tanh NOT WORKING
    out = layer = Dropout(0.5)(layer) #changed to out
    #out = Dense(num_classes, activation='relu')(layer)  #tanh NOT WORKING

    if fusion: return Model(inputs=input_lid, outputs=out)

    if strategy == 'one_hot':
       out = Dense(num_classes,activation='softmax')(layer)
    elif strategy == 'reg':
       out = Dense(num_classes)(layer)
    return Model(inputs = input_lid, outputs = out)


def ResLike(input_shape, num_classes, strategy, fusion= False):
    dropProb = 0.3
    channel = 32  # 32 now is the best, better than 64, 16
    input_lid = Input(shape=input_shape)
    a = layer = Conv2D(channel, kernel_size=(3, 3),
                       activation='relu', padding="SAME", input_shape=input_shape)(input_lid)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) #+ a
    layer = Add()([layer, a]) # DR
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    b = layer = Dropout(dropProb)(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) #+ b
    layer = Add()([layer, b]) # DR
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    c = layer = Dropout(dropProb)(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) #+ c
    layer = Add()([layer, c]) # DR
    layer = MaxPooling2D(pool_size=(1, 2))(layer)
    d = layer = Dropout(dropProb)(layer)

    # if add this layer, need 35 epochs to converge
    # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + d
    # layer = MaxPooling2D(pool_size=(1, 2))(layer)
    # e = layer = Dropout(dropProb)(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) #+ d
    layer = Add()([layer, d]) # DR

    layer = Flatten()(layer)
    layer = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer)
    layer = Dropout(0.2)(layer)  # 0.25 is similar ... could try more values
    layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(layer)
    out = layer = Dropout(0.2)(layer)  # 0.25 is similar ... could try more values
    
    if fusion : return Model(inputs=input_lid, outputs=out)
    if strategy == 'one_hot':
        out = Dense(num_classes, activation='softmax')(layer)
    elif strategy == 'reg':
        out = Dense(num_classes)(layer)
    return Model(inputs=input_lid, outputs=out)