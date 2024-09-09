import getopt, sys
from pickle import load
import random
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import f1_score



from numpy import array
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout

from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model

from sklearn.preprocessing import StandardScaler
import plotly
import plotly.graph_objects as go

import tensorflow as tf

#import warnings

def reproducibleResults(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)



if __name__ == "__main__":
    #warnings.filterwarnings("ignore")

    # read cmd line arguments
    try:
        arguments, values = getopt.getopt(sys.argv[1:], "d:n:m:", ["dataset=", "timeSeriesNum=", "errorVal="])
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    inputFile = 'nasd_query.csv'	# default value
    n = 5				# default value
    threshold = 0.0015				# default value
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-d", "--dataset"):
            inputFile = currentValue
        elif currentArgument in ("-n", "--timeSeriesNum"):
            n = int(currentValue)
        elif currentArgument in ("-m", "--errorVal"):
            errorValue = float(sys.argv[6])
    # end reading cmd line arguments

    # print arguments to test if correct
    print(inputFile)
    print(n)
    print(errorValue)

    ###################Scaler and args ############################################
    scaler = np.load('./scaler_3b.pkl', allow_pickle=True)
    encoder2 =  keras.models.load_model('./autoencoder_3b')
    dataset = inputFile
    TIME_STEPS = 30

    ###############################################################################
    
    #read dataset ,drop null rows and save id column for later
    df_dataset = pd.read_csv(dataset,delimiter='\t',header = None)
    df_dataset.dropna
    ids = df_dataset.iloc[:, 0]
    print(ids)
    df_dataset.drop(df_dataset.columns[0],axis=1, inplace=True)
    df_data = df_dataset.to_numpy()

    print(df_dataset)

    count = 0
    count2 = 0
    for row in df_data:
        
        if count < n:
            #take row id
            id = ids[count]
            #making data   
            row =  row.reshape(row.shape[0],1)
            row = scaler.transform(row)
            
            #create dataset in timestems
            X_row, y_row = create_dataset(row,row,TIME_STEPS)
            y_row.reshape(y_row.shape[0])
            y_row.reshape(y_row.shape[0])

            #passing data through the network
            X_row_pred = encoder2.predict(X_row)
            #plot loss
            count2 = count2+1
            figure(count2,figsize=(8,6),dpi=100)

            test_mae_loss = np.mean(np.abs(X_row_pred - X_row), axis=1)

            plt.hist(test_mae_loss, bins=50)
            plt.xlabel('Test MAE loss')
            plt.ylabel('Number of Samples');
            plt.title('Loss')
            #plt.legend();
            plt.savefig('./B_plots/{}_{}_test_loss.png'.format(count,id))
            #plt.show()

            #plot threshold

            count2 = count2 +1            
            figure(count2,figsize=(8,6),dpi=100)

            test_score_df = pd.DataFrame(row[TIME_STEPS:])
            test_score_df['loss'] = test_mae_loss
            test_score_df['threshold'] = threshold
            test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
            test_score_df['Close'] = row[TIME_STEPS:]
          

            plt.plot(test_score_df['loss'], label='loss')
            plt.plot(test_score_df['threshold'],label = 'threshold')
            plt.title('Thershold')
            plt.ylabel('Test MAE loss')
            plt.xlabel('Number of samples');
            plt.legend();
            plt.savefig('./B_plots/{}_{}_threshold.png'.format(count,id))
            print(row.shape)
            #plt.show()
            
            #find anomalies
            anomalies = test_score_df.loc[test_score_df['anomaly'] == True]
            anomalies.reset_index(drop = True)
            print(anomalies.index)
            anomalies.shape
            data_top =  list(anomalies.index)
            print(data_top)

            #plot anomalies
            if len(anomalies) != 0:

                fig = go.Figure()

                close = test_score_df['Close'].to_numpy().reshape(test_score_df.shape[0],1)
                c=scaler.inverse_transform(close)
                a_close = anomalies['Close'].to_numpy().reshape(anomalies.shape[0],1)
                a = scaler.inverse_transform(a_close)
              
                x = np.arange(len(c))
                x = x.reshape(len(x),1)
              
                test_score_df['Close'] = c
                test_score_df['Dates'] = x

                anomalies.loc[:,['Close']] = a
                anomalies.loc[:,['Dates']] = data_top
                
                count2 = count2 +1            
                figure(count2,figsize=(10,8),dpi=100)

                fig.add_trace(go.Scatter( x = test_score_df['Dates'] ,y=test_score_df['Close'], name='Close price'))
                fig.add_trace(go.Scatter(x=anomalies['Dates'], y= anomalies['Close'], mode='markers', name='Anomaly'))
                fig.update_layout(showlegend=True, title='Detected anomalies')
                fig.write_image('./B_plots/{}_{}_anomalies.png'.format(count,id))

                #fig.show()

        #if row > n break 
        else: 
            break

        #row counter    
        count = count+1    

