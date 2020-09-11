from __future__ import division

import six
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Conv2D, add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential

from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.regularizers import l2
from keras import backend as K

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