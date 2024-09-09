project3

Stefanos Baklavas: 1115201700093

Alexia Topalidou: 1115201600286


Question A: Time Series Prediction using LSTM (Long Short-Term Memory) Model

Initially, for each time series, we train the model and make predictions for the specific time series. Once this process is completed, the model 
trained with all time series is loaded and we make again predictions for each time series. 
Program: forecast.py
Run: ./forecast.py

Question B:  Anomaly detection in time series usnig Autoencoder model

An autoencoder model compresses the time series dataset and attempts to reconstruct it, identifying anomalies that cause issues in the reconstruction 
based on a threshold we set.
Program: detect.py

Run: python3 detect.py -d ./dataset.csv -n 4 -m 0.0015
    n= number of time series
    m= threshold for anomaly detection

Question C:  Compressing  datasets using autoencoder model.

In Question C reduces the dimensionality of two input time series datasets (a dataset and a query set) using a pre-trained autoencoder model. It 
reads the datasets (dataset and query set), compresses them with the autoencoder, and writes the reduced time series to output files.
Program: reduce.py

Run:
python3 reduce.py -d ./dataset.csv -q ./query.csv -od ./compressed_dataset.csv -oq ./compressed_queryset.csv

Contents of this folder:

    Models:
        a. Trained model for Question A: my_model
        b. Trained model for Question B: autoencoder_3b
        c. Trained model for Question C: autoencoder_3c (not used)
        d. Trained model for Question C: encoder_3c

    Colab Notebooks:
        The training and tests for Questions B and C were conducted using Google Colab, and the notebooks are project_3b.ipynb and project_3c.ipynb.

    Python files:
        a. forecast.py, wich is used for both training and loading the best model during the evaluation for Question A.
        b. detect.py, which loads the best model for Question B and generates the plots in the folder B_plots.
        c. reduce.py, which loads the best model for Question C and produces the compressed files.

    Folders:
        outputs_project2 and compressed_outputs_project2 contain the results for all questions of the 2nd project with the original and compressed 
        datasets respectively. To save time, the first 50 time series were used for the dataset and the next 3 for the query set from the provided file.

        Two scalers for Questions B and C, fitted on the entire dataset of 360 time series.

       .csv files
       
     Project Report:
        Project3_report.pdf.
