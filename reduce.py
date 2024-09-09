# $python reduce.py –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>

import argparse
import getopt
import math
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np

from numpy import array
from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout

from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector
from keras.models import Model
from keras.models import model_from_json
from keras import regularizers
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


import random
import os
import tensorflow as tf


#this functions is used if we want to split X in windows
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(0,len(X) - time_steps,time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def windowing(X,window_size=1):
    Xs = []
    for i in range(0,len(X) - window_size,window_size):
        v = X[i:(i + window_size)]
        Xs.append(v)
    return np.array(Xs)




def make_initial_timeseries(X):
    Xs = []
    for i in range(0,X.shape[0]):
      for item in X[i]:
        Xs.append(item)

    return np.array(Xs)


def create_output_files(data,window_length,encoder2,scaler):


    reconstructed_dataset = []
    row_counter = 0
    for row in range(0,data.shape[0]):
        x = data[row]
        x = x.reshape(x.shape[0],1)
        #print(x.shape)
        x = scaler.transform(x)
        x = windowing(x,window_length)
        #print(x.shape)
        compressed = encoder2.predict(x)

        #print(compressed.shape)
        compressed = make_initial_timeseries(compressed)
        #print(compressed.shape)

        compressed_unscaled = scaler.inverse_transform(compressed)
        reconstructed_dataset.append(compressed_unscaled)
        #print(compressed)
        row_counter = row_counter + 1
    print("row_counter = ", row_counter)    
    print("create_output_files = ",data.shape[0])
    print("len_reconstructed = ",len(reconstructed_dataset))

    return reconstructed_dataset    



def write_to_file(filepath,dataset,ids):
  textfile = open(filepath, "w")

  row_counter = 0

  for row in dataset:
      counter1 = 0
      for number in row:          
          if counter1 == 0:
              textfile.write(str(ids[row_counter]) + "\t")    
    
          textfile.write(str(number[0]) + "\t")
          counter1 = counter1 + 1
      
      
      if row_counter != (len(dataset) -1):
          textfile.write("\n")  
      else:
          print("done")
      row_counter = row_counter + 1
  textfile.close()
  print("row_counter = ",row_counter)  

if __name__ == "__main__":


    
    inputdataset = str()
    inputqueryset = str()
    outputdataset = str()
    outputqueryset = str()
   

    args = sys.argv[1:]
    for i in range(0,len(args)):
        if args[i] == "-d":
            inputdataset = args[i+1]
        if args[i] == "-q":
            inputqueryset = args[i+1]
        if args[i] == "-od":
            outputdataset = args[i+1]
        if args[i] == "-oq":
            outputqueryset = args[i+1]

   

    print(inputdataset)
    print(inputqueryset)
    print(outputdataset)
    print(outputqueryset)

    dataset_data = inputdataset
    query_set = inputqueryset
    window_length = 10


    #Read dataset and queryset
    df_dataset_data = pd.read_csv(dataset_data,delimiter = '\t',header = None,nrows=50)
    df_query_set = pd.read_csv(query_set,delimiter = '\t',header = None)
    
    #df_out = pd.read_csv(dataset_data,delimiter = '\t',header = None,skiprows = 50,nrows=3)
    #df_out.to_csv('query_corrected.csv',sep = '\t',index=False,header = None)
    
    #df_out = pd.read_csv(dataset_data,delimiter = '\t',header = None,nrows = 50)
    #df_out.to_csv('dataset_corrected.csv',sep= '\t',index=False,header = None)

    df_dataset_data.dropna
    ids_dataset = df_dataset_data.iloc[:, 0]
    print(ids_dataset[0])
    df_query_set.dropna
    ids_query = df_query_set.iloc[:, 0]



    #drop column with id's
    df_dataset_data.drop(df_dataset_data.columns[0],axis=1, inplace=True)
    df_query_set.drop(df_query_set.columns[0],axis=1, inplace=True)

    
    #load trained model and scaler and turn data to numpy
    scaler = np.load('./scaler_3c.pkl', allow_pickle=True)
    encoder2 =  keras.models.load_model('./encoder_3c')
    data = df_dataset_data.to_numpy()
    print(data[0])
    query = df_query_set.to_numpy()
    print(query.shape)
    #data = scaler.transform(data)

    #reconstruct dataset and query set
    reconstructed_dataset =  create_output_files(data,window_length,encoder2,scaler)
    reconstructed_queryset = create_output_files(query,window_length,encoder2,scaler)

    #creating the new .csv files with reduced dimension timesieries
    write_to_file(outputdataset,reconstructed_dataset,ids_dataset)
    write_to_file(outputqueryset,reconstructed_queryset,ids_query)