import getopt, sys
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
#import warnings


if __name__ == "__main__":
    #warnings.filterwarnings("ignore")

    # read cmd line arguments
    try:
        arguments, values = getopt.getopt(sys.argv[1:], "d:n:", ["dataset=", "timeSeriesNum="])
    except getopt.error as err:
        print (str(err))
        sys.exit(2)

    inputFile = './dataset.csv'	# default value
    numOfTimeSeries = 3				# default value
    for currentArgument, currentValue in arguments:
        if currentArgument in ("-d", "--dataset"):
            inputFile = currentValue
        elif currentArgument in ("-n", "--timeSeriesNum"):
            numOfTimeSeries = currentValue
            numOfTimeSeries = int(numOfTimeSeries)
    # end reading cmd line arguments

    # print arguments to test if correct
    # print(inputFile)
    # print(numOfTimeSeries)

    timeSeriesList = []
    with open(inputFile) as csvfile:
        data = csv.reader(csvfile, delimiter = '\t')
        for x in range(numOfTimeSeries):
            timeSeriesList.append(next(data)) # read first n time series and push in a list
    
    # concatenate first n timeSeries and push 80% of each time series in list as one big time series
    flat_list = []
    for sublist in timeSeriesList:
        del sublist[0] # delete id
        length = len(sublist)
        newLength = length * 0.8
        for item in sublist:
            flat_list.append(item)
            newLength = newLength - 1
            if newLength == 0:
                break
    timeSeriesList.append(flat_list)
    # end of concatenation
    
    line = 0
    for x in range(numOfTimeSeries + 1):
        # del timeSeriesList[line][0] # delete id
        arr = np.array(timeSeriesList[line]) # convert time series to array
        arr = arr.astype('float64') # convert time serie data to float

        if line != (len(timeSeriesList)-1): # if we will train the model with one time series
            # get 80% of array for train data and 20% for test data
            num_of_train_data = (arr.size) * 0.8
            num_of_train_data = int(num_of_train_data)
            # print(num_of_train_data)
            num_of_test_data = (arr.size) * 0.2
            num_of_test_data = int(num_of_test_data)
            # print(num_of_test_data)
            training_set = arr[0:num_of_train_data]
            training_set = np.reshape(training_set, (-1, 1))
            # print(training_set.size)
            test_set = arr[num_of_train_data:arr.size]
            test_set = np.reshape(test_set, (-1, 1))
            # print(test_set.size)

        if line == (len(timeSeriesList)-1): # if we will train the model with all time series
            num_of_train_data = arr.size
            num_of_train_data = int(num_of_train_data)
            training_set = arr[0:num_of_train_data]
            training_set = np.reshape(training_set, (-1, 1))			    

        # Feature Scaling
        sc = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled = sc.fit_transform(training_set)
        # Creating a data structure with 60 time-steps and 1 output
        X_train = []
        y_train = []
        for i in range(60, num_of_train_data):
            X_train.append(training_set_scaled[i-60:i, 0])
            y_train.append(training_set_scaled[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        if line != (len(timeSeriesList)-1): # don't train for n timeseries because it will be loaded
            model = Sequential()
            #Adding the first LSTM layer and some Dropout regularisation
            model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
            model.add(Dropout(0.2))
            # Adding a second LSTM layer and some Dropout regularisation
            model.add(LSTM(units = 50, return_sequences = True))
            model.add(Dropout(0.2))
            # Adding a third LSTM layer and some Dropout regularisation
            model.add(LSTM(units = 50, return_sequences = True))
            model.add(Dropout(0.2))
            # Adding a fourth LSTM layer and some Dropout regularisation
            model.add(LSTM(units = 50))
            model.add(Dropout(0.2))
            # Adding the output layer
            model.add(Dense(units = 1))

            # Compiling the RNN
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')

            # Fitting the RNN to the Training set
            model.fit(X_train, y_train, epochs = 30, batch_size = 32)
            # saving the model
            if line == (len(timeSeriesList)-1):
                model.save("my_model")

        if line != (len(timeSeriesList)-1): # if we trained the model with one time series
            arr = np.reshape(arr, (-1, 1))
            inputs = arr[(arr.size)-num_of_test_data-60:arr.size]
            inputs = np.reshape(inputs, (-1, 1))
            # print(inputs.size)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60, inputs.size):
                X_test.append(inputs[i-60:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            # print(X_test.shape)

            predicted_value = model.predict(X_test)
            predicted_value = sc.inverse_transform(predicted_value)

            time = np.arange(start=num_of_train_data, stop=arr.size, step=1)

            # Visualising the results
            plt.plot(time,test_set, color = 'red', label = 'Real value')
            plt.plot(time,predicted_value, color = 'blue', label = 'Predicted value')
            plt.xticks(np.arange(num_of_train_data,arr.size,50)) #######3
            plt.title('Prediction')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

        if line == (len(timeSeriesList)-1): # if we trained the model with all time series
            lines = 0
            m_model = keras.models.load_model("my_model")
            for x in range(numOfTimeSeries):
                # del timeSeriesList[line][0] # delete id
                arr = np.array(timeSeriesList[lines]) # convert time series to array
                arr = arr.astype('float64') # convert time serie data to float
                num_of_train_data = (arr.size) * 0.8
                num_of_train_data = int(num_of_train_data)		    
                num_of_test_data = (arr.size) * 0.2
                num_of_test_data = int(num_of_test_data)
                training_set = arr[0:num_of_train_data]
                training_set = np.reshape(training_set, (-1, 1))
                # print(training_set.size)
                test_set = arr[num_of_train_data:arr.size]
                test_set = np.reshape(test_set, (-1, 1))			    		    
                arr = np.reshape(arr, (-1, 1))
                inputs = arr[(arr.size)-num_of_test_data-60:arr.size]
                inputs = np.reshape(inputs, (-1, 1))
                # print(inputs.size)
                inputs = sc.transform(inputs)
                X_test = []
                for i in range(60, inputs.size):
                    X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                # print(X_test.shape)

                predicted_value = m_model.predict(X_test)
                predicted_value = sc.inverse_transform(predicted_value)

                time = np.arange(start=num_of_train_data, stop=arr.size, step=1)

                # Visualising the results
                plt.plot(time,test_set, color = 'red', label = 'Real value')
                plt.plot(time,predicted_value, color = 'blue', label = 'Predicted value')
                plt.xticks(np.arange(num_of_train_data,arr.size,50)) #######3
                plt.title('Prediction')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.show()

                lines = lines + 1

        line = line + 1