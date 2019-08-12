#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: eleftherios
@github: https://github.com/trivizakis

"""

from data_generator import DataGenerator
from model import CustomModel
from dataset import DataConverter
from utils import Utils
from pathlib import Path
import numpy as np
import pickle as pkl
import keras
from keras import backend as K
from keras import Model

version="5"

#path = "chkp/network_version:"+version+"/"
path = "chkp/MIAS 2-class test/"

model = path+version+".h5"
pids_path = path+"test"+version+".npy"
#pids_path = path+"pids_tr"+version+".npy"
#pids_path = path+"pids_val"+version+".npy"
hypes_path = path+"hypes"

hyperparameters = Utils.get_hypes(path=hypes_path)

#create npy files and labels
my_file = Path(hyperparameters["dataset_dir"] + "labels.pkl")
if not my_file.is_file():
    DataConverter(hyperparameters).convert_png_to_npy()

#image-CNN
pids_test=np.load(pids_path)
pids=np.load(hyperparameters["dataset_dir"] + "pids.npy")
with open(hyperparameters["dataset_dir"]+"labels.pkl", "rb") as file:
    labels = pkl.load(file)

#correct labels
label_values = np.load(hyperparameters["dataset_dir"] + "labels.npy")

#remove zero class (DDSM) and/or re-value other classes (eg for different analyses)
ones=0
twos=0
threes=0
fours=0
others=0
label=label_values.tolist()
for index in range(0,len(label)):
    #for mini-MIAS 0 is fatty
    #1 can be zero (glandular) for two-class
    #2 can be one (dense) for two-class
    # no 3 & 4
    if label[index] == 1:
        label_values[index]=0
        labels[pids[index]]=0
        ones+=1
    elif label[index] == 2:
        label_values[index]=1
        labels[pids[index]]=1
        twos+=1
    elif label[index] == 3:
        label_values[index]=1
        labels[pids[index]]=1
        threes+=1
    elif label[index] == 4:
        label_values[index]=1
        labels[pids[index]]=1
        fours+=1
    else:#like 0
#        print(label[index])
#        print(pids[index])
#        print(index)
        others+=1
print(others)
print("MIN L: "+str(label_values.min()))
print("MAX L: "+str(label_values.max()))
    

#generator
testing_generator = DataGenerator(pids_test, labels, hyperparameters, training=False)

#clear session in every iteration        
K.clear_session()

#create network
cnn = CustomModel.get_model(hyperparameters)

cnn.load_weights(model)

#image CNN - testing set performance
CustomModel.test_model(cnn,hyperparameters,testing_generator)

#native Keras evaluation
#score = cnn.evaluate_generator(generator=testing_generator, steps=1, callbacks=Utils.get_callbacks(hyperparameters), max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
#score = cnn.evaluate_generator(testing_generator)
#print(cnn.metrics_names)
#print(score)

#plot Keras model
#from keras.utils import plot_model
#plot_model(cnn, to_file='model.png')

#keras.utils.plot_model(cnn, to_file="model1_png",show_shapes=False, show_layer_names=True)
