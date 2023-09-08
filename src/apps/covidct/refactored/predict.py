#!/usr/bin/env python3
#
# to predict using GPU we need to install a different tensorflow:
#  eg. tensorflow-gpu==1.14.0
# this is the oldest
#   tensorflow==1.14.0  keras==2.2.5
# and newest:
#   tensorflow==2.1.0 keras==2.3.1
# current is :
#   keras==2.2.5 and tensorflow from /disk/zhitingz/serverless-gpus/build/tensorflow_cudart_dynam/build/
#   which you install with python3 -m pip install <path to .whl>

import glob
import os
import numpy as np

GPU = True if "USE_GPU" in os.environ and os.environ["USE_GPU"] == "1" else False
TF_GPU = True if "TF_GPU" in os.environ and os.environ["TF_GPU"] == "1" else False

#TODO: fix this hack. it's for using gpu 3 on host
if TF_GPU or GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

if not GPU:
    from skimage.transform import resize
if GPU:
    import cupy as cp
    cp.cuda.Device(0).use()
    from cucim.skimage.transform import resize

#disable these annoying messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# don't allocate everything
# this is for tensorflow v2
# physical_devices = tf.config.list_physical_devices('GPU') 
# for gpu_instance in physical_devices: 
#     tf.config.experimental.set_memory_growth(gpu_instance, True)

#this works for tf v1
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.intra_op_parallelism_threads = 1
config.inter_op_parallelism_threads = 1
config.use_per_session_threads = 1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ConvLSTM2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D,Dropout
from keras.utils import to_categorical
import h5py
from timer import Timer

#globals
BCDU_MODEL = None
CNN_MODEL = None

def BCDU_net_D3(input_size = (128,128,1)):
    N = input_size[0]
    inputs = Input(input_size) 
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)
    
    up6 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 128))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 128))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal')(merge6)
            
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 64))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 64))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 32))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 32))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 16, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs, conv9)
    return model

def get_BCDU_loaded(path_to_model):
    model = BCDU_net_D3(input_size = (128,128,1))
    model.load_weights(path_to_model)
    return model

def get_cnn_model(path_to_model):
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(50,128,128,1)))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform',padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Flatten())
    model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.3))
    model.add(Dense(3, activation='softmax'))

    model.load_weights(path_to_model)
    return model
    
def main(input_dir, bcdu_model_path, cnn_model_path):
    global BCDU_MODEL, CNN_MODEL

    with Timer.get_handle("load_bcdu"):
        if BCDU_MODEL is None:
            BCDU_MODEL = get_BCDU_loaded(bcdu_model_path)

    file_paths = glob.glob(os.path.join(input_dir, '*.npy'))

    if not GPU:
        dataset = np.zeros((len(file_paths),50,128,128))
    else:
        dataset = cp.zeros((len(file_paths),50,128,128))
    
    counter = 0
    for j in file_paths:
        if not GPU:
            CT = np.load(j)
        else:
            CT = cp.load(j)
        with Timer.get_handle("resize_128_128_#1"):
            CT_resized = resize(CT, (CT.shape[0],128, 128), anti_aliasing=True)

        with Timer.get_handle("bcdu_predict"):
            if not GPU:
                out = BCDU_MODEL.predict(np.reshape(CT_resized,(CT_resized.shape[0],CT_resized.shape[1],CT_resized.shape[2],1)))
                c = CT_resized-out[:,:,:,0]
            else:
                reshaped = cp.reshape(CT_resized,(CT_resized.shape[0],CT_resized.shape[1],CT_resized.shape[2],1))
                out = BCDU_MODEL.predict(reshaped.get())
                out2 = cp.asarray(out)
                c = CT_resized - out2[:,:,:,0]

        with Timer.get_handle("resize_128_128_#2"):
            dataset[counter] = resize(c,(50,128,128))
        counter +=1

    with Timer.get_handle("reshape"):
        if not GPU:
            dataset = np.reshape(dataset,(dataset.shape[0],dataset.shape[1],dataset.shape[2],dataset.shape[3],1))
        else:
            dataset = cp.reshape(dataset,(dataset.shape[0],dataset.shape[1],dataset.shape[2],dataset.shape[3],1))

    with Timer.get_handle("load_cnn"):
        if CNN_MODEL is None:
            CNN_MODEL = get_cnn_model(cnn_model_path)

    #with tf.device('/gpu:3'):
    with Timer.get_handle("predict"):
        if not GPU:
            x = CNN_MODEL.predict(dataset)
        else:
            x = CNN_MODEL.predict(dataset.get())

    print(x)

if __name__ == "__main__":
    with Timer.get_handle(f"end-to-end w/ load {'GPU' if GPU else 'CPU'}"):
        input_dir = "./output2"
        bcdu_model = os.path.join("./models", "weight_bcdunet_v2.hdf5")
        cnn_model = os.path.join("./models", "weight_cnn_CovidCtNet_v2_final.h5")
        main(input_dir, bcdu_model, cnn_model)
    Timer.print()

    # Timer.reset()
    # for _ in range(3):
    #     with Timer.get_handle(f"loaded end-to-end {'GPU' if GPU else 'CPU'}"):
    #         input_dir = "./output2"
    #         bcdu_model = os.path.join("./models", "weight_bcdunet_v2.hdf5")
    #         cnn_model = os.path.join("./models", "weight_cnn_CovidCtNet_v2_final.h5")
    #         main(input_dir, bcdu_model, cnn_model)
    # Timer.print(ignore_first=1)
