'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - DNN training
Restructured Feb 18, 2020 - UTI to STFT
Restructured ..., 2020 - UTI to STFT 3D-CNN  by Laszlo Toth <tothl@inf.u-szeged.hu>
Restructured April 15, 2021 - UTI to STFT-Tacotron2 by Csaba Zainkó <zainko@tmit.bme.hu>


Keras implementation of the UTI-to-STFT-Tacotrn2 model of
Implementation of Csaba Zainkó, László Tóth, Amin Honarmandi Shandiz, Gábor Gosztolya, Alexandra Markó, Géza Németh, Tamás Gábor Csapó, 
,,Adaptation of Tacotron2-based Text-To-Speech for Articulatory-to-Acoustic Mapping using Ultrasound Tongue Imaging'', submitted to SSW11
'''
import sys
print(sys.version)
import tensorflow as tf
print(tf.__version__)

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import gc
import re
import tgt
import csv
import datetime
import scipy
import pickle
import random
random.seed(17)
import skimage

import tensorflow.keras as keras
import tensorflow.keras
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Flatten, Activation, InputLayer, Dropout, Lambda,Softmax

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
# additional requirement: SPTK 3.8 or above in PATH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# do not use all GPU memory
from tensorflow.keras.backend import set_session
config =tf.ConfigProto()
config.gpu_options.allow_growth = True 
set_session(tf.Session(config=config))


sts = 6
window_size = sts*4+1
n_to_skip = np.floor(window_size // 2).astype(np.int64)


#defining the swish activation function
from tensorflow.keras import backend as K

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)


tf.keras.utils.get_custom_objects().update({'swish': Swish(swish)})

def strided_app(a, L, S, verbose=None):  # Window len = L, Stride len/stepsize = S
    shape = a.shape[1:]
    nrows = ((a.shape[0] - L) // S) + 1
    strides = a.strides
    #print(shape, strides)
    if verbose:
        print("strides:", strides)
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L) + shape,
                                           strides=(S * strides[0],) + strides) 


# read_ult reads in *.ult file from AAA
def read_ult(filename, NumVectors, PixPerVector):
    # read binary file
    ult_data = np.fromfile(filename, dtype='uint8')
    ult_data = np.reshape(ult_data, (-1, NumVectors, PixPerVector))
    return ult_data


# read_psync_and_correct_ult reads *_sync.wav and finds the rising edge of the pulses
# if there was a '3 pulses bug' during the recording,
# it removes the first three frames from the ultrasound data
def read_psync_and_correct_ult(filename, ult_data):
    (Fs, sync_data_orig) = io_wav.read(filename)
    sync_data = sync_data_orig.copy()

    # clip
    sync_threshold = np.max(sync_data) * 0.6
    for s in range(len(sync_data)):
        if sync_data[s] > sync_threshold:
            sync_data[s] = sync_threshold

    # find peeks
    peakind1 = detect_peaks(sync_data, mph=0.9*sync_threshold, mpd=10, threshold=0, edge='rising')
    
    # this is a know bug: there are three pulses, after which there is a 2-300 ms silence, 
    # and the pulses continue again
    if (np.abs( (peakind1[3] - peakind1[2]) - (peakind1[2] - peakind1[1]) ) / Fs) > 0.2:
        bug_log = 'first 3 pulses omitted from sync and ultrasound data: ' + \
            str(peakind1[0] / Fs) + 's, ' + str(peakind1[1] / Fs) + 's, ' + str(peakind1[2] / Fs) + 's'
        print(bug_log)
        
        peakind1 = peakind1[3:]
        ult_data = ult_data[3:]
    
    for i in range(1, len(peakind1) - 2):
        # if there is a significant difference between peak distances, raise error
        if np.abs( (peakind1[i + 2] - peakind1[i + 1]) - (peakind1[i + 1] - peakind1[i]) ) > 1:
            bug_log = 'pulse locations: ' + str(peakind1[i]) + ', ' + str(peakind1[i + 1]) + ', ' +  str(peakind1[i + 2])
            print(bug_log)
            bug_log = 'distances: ' + str(peakind1[i + 1] - peakind1[i]) + ', ' + str(peakind1[i + 2] - peakind1[i + 1])
            print(bug_log)
            
            raise ValueError('pulse sync data contains wrong pulses, check it manually!')
    return ([p for p in peakind1], ult_data)


def get_training_data(dir_file, filename_no_ext, NumVectors = 64, PixPerVector = 842):
    print('starting ' + dir_file + filename_no_ext)

    # read in raw ultrasound data
    ult_data = read_ult(dir_file + filename_no_ext + '.ult', NumVectors, PixPerVector)
    
    try:
        # read pulse sync data (and correct ult_data if necessary)
        (psync_data, ult_data) = read_psync_and_correct_ult(dir_file + filename_no_ext + '_sync.wav', ult_data)
    except ValueError as e:
        raise
    else:
        
        # works only with 22kHz sampled wav
        (Fs, speech_wav_data) = io_wav.read(dir_file + filename_no_ext + '_speech_volnorm.wav')
        assert Fs == 22050

        mgc_lsp_coeff = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.mgclsp', dtype=np.float32).reshape(-1, order + 1)
        lf0 = np.fromfile(dir_file + filename_no_ext + '_speech_volnorm_cut_ultrasound.lf0', dtype=np.float32)

        (mgc_lsp_coeff_length, _) = mgc_lsp_coeff.shape
        (lf0_length, ) = lf0.shape
        assert mgc_lsp_coeff_length == lf0_length

        # cut from ultrasound the part where there are mgc/lf0 frames
        ult_data = ult_data[0 : mgc_lsp_coeff_length]

        # read phones from TextGrid
        tg = tgt.io.read_textgrid(dir_file + filename_no_ext + '_speech.TextGrid')
        tier = tg.get_tier_by_name(tg.get_tier_names()[0])

        tg_index = 0
        phone_text = []
        for i in range(len(psync_data)):
            # get times from pulse synchronization signal
            time = psync_data[i] / Fs

            # get current textgrid text
            if (tier[tg_index].end_time < time) and (tg_index < len(tier) - 1):
                tg_index = tg_index + 1
            phone_text += [tier[tg_index].text]

        # add last elements to phone list if necessary
        while len(phone_text) < lf0_length:
            phone_text += [phone_text[:-1]]

        print('finished ' + dir_file + filename_no_ext + ', altogether ' + str(lf0_length) + ' frames')

        return (ult_data, mgc_lsp_coeff, lf0, phone_text)

#read target vectors
def get_vec_data(filename):
    d=np.load(filename)
    return d.astype(np.float32)

# Parameters of old vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
n_mgc = order + 1

#vector
n_vecspec=93

# parameters of ultrasound images
framesPerSec = 81.67
type = 'PPBA' # the 'PPBA' directory can be used for training data
# type = 'EszakiSzel_1_normal' # the 'PPBA' directory can be used for test data
n_lines = 64
n_pixels = 842
n_pixels_reduced = 128


# TODO: modify this according to your data path
dir_base = "../data_SSI2018/"
##### training data
# - 2 females: spkr048, spkr049
# - 5 males: spkr010, spkr102, spkr103, spkr104, spkr120
# speakers = ['spkr048', 'spkr049', 'spkr102', 'spkr103']
speakers = ['spkr048']

for speaker in speakers:
    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + "/" + type + "/"
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            if file.endswith('_speech_volnorm_cut_ultrasound.mgclsp'):
                ult_files_all += [dir_data + file[:-37]]
    
    # randomize the order of files
    random.shuffle(ult_files_all)
    # temp: only first 10 sentence
    #ult_files_all = ult_files_all[0:10]
    
    ult_files = dict()
    ult = dict()
    vecspec = dict()
    ultmel_size = dict()
   

    # train: first 90% of sentences
    ult_files['train'] = ult_files_all[0:int(0.9*len(ult_files_all))]
    # valid: last 10% of sentences
    ult_files['valid'] = ult_files_all[int(0.9*len(ult_files_all)):]
    
    for train_valid in ['train', 'valid']:
        n_max_ultrasound_frames = len(ult_files[train_valid]) * 500
        ult[train_valid] = np.empty((n_max_ultrasound_frames, n_lines, n_pixels_reduced))
        vecspec[train_valid] = np.empty((n_max_ultrasound_frames, n_vecspec))
        ultmel_size[train_valid] = 0
        
        # load all training/validation data
        for basefile in ult_files[train_valid]:
            try:
                (ult_data, mgc_lsp_coeff, lf0, phone_text) = get_training_data('', basefile)
                
                vec_data = get_vec_data(basefile + '.onehot.npy')
                
                
                
                
            except ValueError as e:
                print("wrong psync data, check manually!", e)
            else:
                ultmel_len = np.min((len(ult_data),len(vec_data)))
                ult_data = ult_data[0:ultmel_len]
                vec_data = vec_data[0:ultmel_len]
                
                print(basefile, ult_data.shape, vec_data.shape)
                
                if ultmel_size[train_valid] + ultmel_len > n_max_ultrasound_frames:
                    print('data too large', n_max_ultrasound_frames, ultmel_size[train_valid], ultmel_len)
                    raise
                
                for i in range(ultmel_len):
                    ult[train_valid][ultmel_size[train_valid] + i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
                
                vecspec[train_valid][ultmel_size[train_valid] : ultmel_size[train_valid] + ultmel_len] = vec_data
                ultmel_size[train_valid] += ultmel_len
                
                print('n_frames_all: ', ultmel_size[train_valid])


        ult[train_valid] = ult[train_valid][0 : ultmel_size[train_valid]]
        vecspec[train_valid] = vecspec[train_valid][0 : ultmel_size[train_valid]]

        # input: already scaled to [0,1] range
        ult[train_valid] -= 0.5
        ult[train_valid] *= 2
        # reshape ult for CNN
        ult[train_valid] = np.reshape(ult[train_valid], (-1, n_lines, n_pixels_reduced, 1))
        
    print(ult['train'].shape)
    
    #conversion to 3D blocks    
    ult['train'] = strided_app(ult['train'], window_size, 1)
    ult['valid'] = strided_app(ult['valid'], window_size, 1)
    vecspec['train'] = vecspec['train'][n_to_skip:(vecspec['train'].shape[0] - n_to_skip)] 
    vecspec['valid'] = vecspec['valid'][n_to_skip:(vecspec['valid'].shape[0] - n_to_skip)] 

    #Load pretraind model
    melspec_model=keras.models.load_model('./model_melo/UTI_to_STFT_CNN-3D-Laci_spkr048_2021-04-22_06-37-02_weights_best.h5')
                                             
    from tensorflow.keras import layers, models
    
    #reconstract the model to a functional model
    input_layer = layers.Input(batch_shape=melspec_model.layers[0].input_shape)
    prev_layer = input_layer
    for layer in melspec_model.layers:
        if layer.name=='dense': #Only the dense layer is trainable
            layer.trainable=True
        else:
            layer.trainable=False
        
        if layer.name!='dense_1': #Skip the last dense layer.
            prev_layer = layer(prev_layer)
    #new layers at the end            
    x=layers.Dense(n_vecspec,activation='linear',name='dense_2')(prev_layer)
    x=layers.Softmax()(x)
    
    #Functional model
    model = models.Model([input_layer], [x])
    
    #Adam optimzer and categorical crossentropy loss
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc'])
    
    # early stopping to avoid over-training
    earlystopper = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=7, verbose=1, mode='max')
    
    print(model.summary())

    # csv logger
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    print(current_date)
    model_name = 'models/UTI_to_STFT_CNN-3D-ZCS_' + speaker + '_' + current_date
    logger = CSVLogger(model_name + '.csv', append=True, separator=';')

    checkp = ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # save model
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

    #Handling inbalanced classes
    from class_stat import class_weight
    # Run training
    history = model.fit(ult['train'], vecspec['train'],
                            epochs = 140, batch_size = 128, shuffle = True, verbose = 1,
                            validation_data=(ult['valid'], vecspec['valid']),
                            callbacks=[earlystopper, logger,checkp],class_weight=class_weight)
    # here the training of the DNN is finished



