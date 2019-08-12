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
from pandas import read_table as rt

version="1"

#path = "chkp/network_version:"+version+"/"
path = "chkp/DDSM 2-class test/"

model = path+version+".h5"
pids_path = path+"test"+version+".npy"
#pids_path = path+"pids_tr"+version+".npy"
#pids_path = path+"pids_val"+version+".npy"
hypes_path = path+"hypes"

hyperparameters = Utils.get_hypes(path=hypes_path)

with open("dataset/DDSM/labels.txt", "rb") as file:
    labels_df=rt(file)

#remove the only zero-labeled subject
labels_df=labels_df.drop(2287)# class 0
labels_df=labels_df.drop(2288)# class 0

labels = dict(labels_df.values)
pids = np.array(list(labels_df['file_name'].values))
label_values = np.array(list(labels_df['label'].values))
pids_test = np.load(pids_path)

#remove zero class and re-value other classes
ones=0
twos=0
threes=0
fours=0
others=0
label=label_values.tolist()
for index in range(0,len(label)):
    #for DDSM 0 is one subject
    #1 can be zero for two-class (fatty)
    #2 can be zero for two-class (glandular)
    # 3 & 4 can be 1 for two-class (dense & extremely dense)
    if label[index] == 1:
        label_values[index]=0
        labels[pids[index]]=0
        ones+=1
    elif label[index] == 2:
        label_values[index]=0
        labels[pids[index]]=0
        twos+=1
    elif label[index] == 3:
        label_values[index]=1
        labels[pids[index]]=1
        threes+=1
    elif label[index] == 4:
        label_values[index]=1
        labels[pids[index]]=1
        fours+=1
    else:
        print(label[index])
        print(pids[index])
        print(index)
        others+=1
print("MIN L: "+str(label_values.min()))
print("MAX L: "+str(label_values.max()))

#input images to npy for generator
my_file = Path(hyperparameters["dataset_dir"]+"pids.npy")
if not my_file.is_file():
    DataConverter(hyperparameters).convert_png_to_npy_with_dict(pids,label_values)
    
#generator
testing_generator = DataGenerator(pids_test, labels, hyperparameters, training=False)

#clear session in every iteration        
K.clear_session()

#create network
cnn = CustomModel.get_model(hyperparameters)

cnn.load_weights(model)

#image CNN - test set performance
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
