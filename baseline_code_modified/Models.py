from __future__ import division

import six
from tensorflow.keras.layers import Activation, BatchNormalization, LeakyReLU, Conv2D, Add,\
    Flatten, MaxPooling2D, Dense, Reshape, Input, Dropout, Concatenate, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers


def Baseline(input_shape, num_classes, strategy):
    dropProb=0.3
    input_lid = Input(shape = input_shape, name='img_input')
    layer = Conv2D(16,kernel_size=(7,7),
                   activation='relu',padding="SAME",input_shape=input_shape, name='img_conv1')(input_lid)
    layer = Conv2D(16, (5, 5), padding="SAME", activation='relu', name='img_conv2')(layer)
    layer = Conv2D(16, (5, 5), padding="SAME", activation='relu', name='img_conv3')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='img_maxpool1')(layer)
    layer = Dropout(dropProb, name='img_dropout1')(layer)
    
    layer = Conv2D(32, (3, 3), padding="SAME", activation='relu', name='img_conv4')(layer)
    layer = Conv2D(32, (3, 3), padding="SAME", activation='relu', name='img_conv5')(layer)
    layer = MaxPooling2D(pool_size=(2, 2), name='img_maxpool2')(layer)
    layer = Dropout(dropProb, name='img_dropout2')(layer)
    
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu', name='img_conv6')(layer)
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu', name='img_conv7')(layer)
    layer = MaxPooling2D(pool_size=(1, 2), name='img_maxpool3')(layer)
    layer = Dropout(dropProb, name='img_dropout3')(layer)
    
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu', name='img_conv8')(layer)
    layer = Conv2D(64, (3, 3), padding="SAME", activation='relu', name='img_conv9')(layer)
    
    layer = Flatten( name='img_flatten')(layer)
    layer = Dense(1024, activation='relu', name='img_dense1')(layer)
    layer = Dropout(0.25, name='img_dropout4')(layer)
    layer = Dense(512, activation='relu', name='img_dense2')(layer)
    layer = Dropout(0.25, name='img_dropout5')(layer)

    if strategy == 'one_hot':
        out = Dense(num_classes,activation='softmax')(layer)
    elif strategy == 'reg':
        out = Dense(num_classes)(layer)
    return Model(inputs = input_lid, outputs = out)    

def BaselineV2(input_shape, num_classes, strategy):
    dropProb=0.25
    channel = 32
    input_lid = Input(shape = input_shape, name='img_input')
    layer1 = Conv2D(channel,kernel_size=(7,7),
                   activation='relu',padding="SAME",input_shape=input_shape, name='img_conv11')(input_lid)
    layer2 = Conv2D(channel,kernel_size=(11,11),
                   activation='relu',padding="SAME",input_shape=input_shape, name='img_conv12')(input_lid)
    layer3 = Conv2D(channel,kernel_size=(3,3),
                   activation='relu',padding="SAME",input_shape=input_shape, name='img_conv13')(input_lid)
    
    layer = Concatenate()([layer1,layer2,layer3])
    layer = MaxPooling2D(pool_size=(2, 2), name='img_maxpool1')(layer)
    layer = Dropout(dropProb, name='img_dropout1')(layer)
    
    
    b = layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv3')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv4')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv5')(layer)  # + b
    layer = Add(name='img_add2')([layer, b])  # DR
    layer = MaxPooling2D(pool_size=(2, 2), name='img_maxpool2')(layer)
    c = layer = Dropout(dropProb, name='img_dropout2')(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv6')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='img_conv7')(layer)  # + c
    layer = Add(name='img_add3')([layer, c])  # DR
    layer = MaxPooling2D(pool_size=(1, 2), name='img_maxpool3')(layer)
    d = layer = Dropout(dropProb, name='img_dropout3')(layer)
    
    layer = Flatten( name='img_flatten')(layer)
    layer = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name='img_dense1')(layer)
    layer = Dropout(0.25, name='img_dropout4')(layer)
    layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name='img_dense2')(layer)
    layer = Dropout(0.25, name='img_dropout5')(layer)

    if strategy == 'one_hot':
        out = Dense(num_classes,activation='softmax')(layer)
    elif strategy == 'reg':
        out = Dense(num_classes)(layer)
    return Model(inputs = input_lid, outputs = out)

def ResLike(input_shape, num_classes, strategy):
    dropProb = 0.3
    channel = 32  # 32 now is the best, better than 64, 16
    input_lid = Input(shape=input_shape, name='lidar_input')
    a = layer = Conv2D(channel, kernel_size=(3, 3),
                       activation='relu', padding="SAME", input_shape=input_shape, name='lidar_conv1')(input_lid)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv2')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv3')(layer)  # + a
    layer = Add(name='lidar_add1')([layer, a])  # DR
    layer = MaxPooling2D(pool_size=(2, 2), name='lidar_maxpool1')(layer)
    b = layer = Dropout(dropProb, name='lidar_dropout1')(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv4')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv5')(layer)  # + b
    layer = Add(name='lidar_add2')([layer, b])  # DR
    layer = MaxPooling2D(pool_size=(2, 2), name='lidar_maxpool2')(layer)
    c = layer = Dropout(dropProb, name='lidar_dropout2')(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv6')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv7')(layer)  # + c
    layer = Add(name='lidar_add3')([layer, c])  # DR
    layer = MaxPooling2D(pool_size=(1, 2), name='lidar_maxpool3')(layer)
    d = layer = Dropout(dropProb, name='lidar_dropout3')(layer)

    # if add this layer, need 35 epochs to converge
    # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer)
    # layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu')(layer) + d
    # layer = MaxPooling2D(pool_size=(1, 2))(layer)
    # e = layer = Dropout(dropProb)(layer)

    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv8')(layer)
    layer = Conv2D(channel, (3, 3), padding="SAME", activation='relu', name='lidar_conv9')(layer)  # + d
    layer = Add(name='lidar_add4')([layer, d])  # DR

    layer = Flatten(name='lidar_flatten')(layer)
    layer = Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense1")(layer)
    layer = Dropout(0.2, name='lidar_dropout4')(layer)  # 0.25 is similar ... could try more values
    layer = Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), name="lidar_dense2")(
        layer)
    out = layer = Dropout(0.2, name='lidar_dropout5')(layer)  # 0.25 is similar ... could try more values

    if strategy == 'one_hot':
        out = Dense(num_classes, activation='softmax')(layer)
    elif strategy == 'reg':
        out = Dense(num_classes)(layer)
    return Model(inputs=input_lid, outputs=out)