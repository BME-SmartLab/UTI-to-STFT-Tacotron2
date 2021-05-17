'''
Written by Tamas Gabor Csapo <csapot@tmit.bme.hu>
First version Nov 9, 2016
Restructured Feb 4, 2018 - get data
Restructured Sep 19, 2018 - DNN training
Restructured Oct 13, 2018 - DNN training
Restructured Feb 18, 2020 - UTI to STFT
Restructured ..., 2020 - UTI to STFT 3D-CNN  by Laszlo Toth <tothl@inf.u-szeged.hu>
Restructured April 15, 2021 - UTI to STFT-Tacotron2 by Csaba Zainkó <zainko@tmit.bme.hu>

Keras implementation of the UTI-to-STFT-Tacotron2 model of
Implementation of Csaba Zainkó, László Tóth, Amin Honarmandi Shandiz, Gábor Gosztolya, Alexandra Markó, Géza Németh, Tamás Gábor Csapó, 
,,Adaptation of Tacotron2-based Text-To-Speech for Articulatory-to-Acoustic Mapping using Ultrasound Tongue Imaging'', submitted to SSW11'''

import matplotlib
matplotlib.use('agg')

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
from detect_peaks import detect_peaks
import os
import os.path
import glob
import tgt
import pickle
import skimage

from scipy.signal import savgol_filter

# sample from Csaba
import WaveGlow_functions

import tensorflow.keras as keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Activation


# do not use all GPU memory
import tensorflow as tf
from tensorflow.keras.backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
try:
    tf_session=tf.Session(config=config)
    set_session(tf_session)

except ValueError as e:
        raise
sts = 6
window_size = sts*4+1
n_to_skip = np.floor(window_size // 2).astype(np.int64)

# defining the swish activation function
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


# Parameters of old vocoder
frameLength = 512 # 23 ms at 22050 Hz sampling
frameShift = 270 # 12 ms at 22050 Hz sampling, correspondong to 81.5 fps (ultrasound)
order = 24
n_mgc = order + 1


# parameters of ultrasound images
framesPerSec = 81.67
# type = 'PPBA' # the 'PPBA' directory can be used for training data
type = 'EszakiSzel_1_normal' # the 'PPBA' directory can be used for test data
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
# speakers = ['spkr102']

# load waveglow model
import soundfile as sf
import sys

for speaker in speakers:
    
    dir_synth = 'synthesized/' + speaker + '/'
    
    # collect all possible ult files
    ult_files_all = []
    dir_data = dir_base + speaker + "/" + type + "/"
    if os.path.isdir(dir_data):
        for file in sorted(os.listdir(dir_data)):
            if file.endswith('_speech_volnorm_cut_ultrasound.mgclsp'):
                ult_files_all += [dir_data + file[:-37]]
    
    csv_files = glob.glob('model_sym/UTI_to_STFT_CNN-3D-ZCS_' + speaker + '_*.csv')
    csv_files = sorted(csv_files)
    melspec_model_name = csv_files[-1][:-4]
    print(csv_files, melspec_model_name)
    
    # melspec network
    with open(melspec_model_name + '_model.json', "r") as json_file:
        loaded_model_json = json_file.read()

    melspec_model=keras.models.load_model(melspec_model_name + '_weights_best.h5')

    print(len(ult_files_all))
    for basefile in ult_files_all:
        try:
            (ult_data, mgc_lsp_coeff, lf0, phone_text) = get_training_data('', basefile)
            
        except ValueError as e:
            print("wrong psync data, check manually!", e)
        else:
            ultmel_len = len(ult_data)
            ult_data = ult_data[0:ultmel_len]
            #mel_data = mel_data[0:ultmel_len]
            
            print('predicting ', basefile, ult_data.shape)
            
            ult_test = np.empty((len(ult_data), n_lines, n_pixels_reduced))
            for i in range(ultmel_len):
                ult_test[i] = skimage.transform.resize(ult_data[i], (n_lines, n_pixels_reduced), preserve_range=True) / 255
            
            # input: scale to [-1, 1]
            ult_test -= 0.5
            ult_test *= 2
            
            # for CNN
            ult_test = np.reshape(ult_test, (-1, n_lines, n_pixels_reduced, 1))
            
            # conversion to 3D blocks    
            ult_test = strided_app(ult_test, window_size, 1)
            
            # predict with the trained DNN
            vecspec_predicted = melspec_model.predict(ult_test)
            
            #Save pure predicted data in two formats:
            print(vecspec_predicted.shape)
            
            output_filename = dir_synth + os.path.basename(basefile) + '.gen_emb_tac.npy'
            np.save(output_filename,vecspec_predicted)
            
            output_filename = dir_synth + os.path.basename(basefile) + '.gen_emb_tac.txt'
            np.savetxt(output_filename,vecspec_predicted,fmt='%3.2f')
            
            
                
            