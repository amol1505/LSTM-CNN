# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 17:08:44 2021

@author: amold
"""

#libraries used to read stock data from yahoo finance 
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
import time 
import requests
import matplotlib.pyplot as plt

from matplotlib.pylab import rcParams
import seaborn as sns
import tensorflow
from pandas_datareader.data import DataReader
from tensorflow.keras.utils import plot_model
#temporary stock ticker which can be manipulated to extract data of desired company
stockticker= "TSLA"

#getting data from yahoo finance from 2010 june all the way to 2020 august 28th
start_point = datetime.datetime(2010, 6, 29)
end_point = datetime.datetime(2020,8, 31)

#creating dataframe for the stock
stock_df = pd.DataFrame()

stock=[]
stock = yf.download(stockticker, start=start_point, end=end_point, progress=False)
stock_df = stock_df.append(stock, sort=False)

stock_df['Symbol'] = stockticker
#displays data during years for tesla from yahoo finance
stock_df

#moving average set at intervals for 10, 30, 60 days for visualisation at a later point

ma_intervals = [10, 30, 60]

for ma in ma_intervals:
    column_name = f"Moving average for {ma} days"
    stock_df2 = stock_df
    stock_df2[column_name] = stock_df2['Adj Close'].rolling(ma).mean()

stock_df2

#training and testing data sets
data= stock_df.filter(['Close'])

#converting the data frame to numpy array
dataset = data.values

#calculating rows to train the model on, ideally
training_datarows = int(np.ceil(len(dataset) * .80))
# returns number of rows
training_datarows

#scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#creating training data set
train_dataset = scaled_data[0:int(training_datarows), :]

#splitting data into x and y for training data sets
X_train = []
Y_train = []

#sliding window approach
for i in range(60, len(train_dataset)):
    X_train.append(train_dataset[i-60:i, 0])
    Y_train.append(train_dataset[i,0])
    
#converting to numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#reshaping data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
from keras.models import Sequential 
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

#model definition
model = Sequential()
model.add(Conv1D(128, kernel_size=1, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(25, activation="relu"))
model.add(Dense(1))
plot_model(model, to_file='model2.png', show_shapes=True)
#defining optimizer and loss function
model.compile(optimizer='adam', loss='mean_squared_error')

#plotting learning curve
history = model.fit(X_train, Y_train, batch_size=1, epochs=4)
print(history.history.keys())
plt.plot(history.history['loss'])
plt.title('CNN Loss over training')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

#test set to validate results
test_data = scaled_data[training_datarows-60: , :]

X_test = []
Y_test = dataset[training_datarows:, :]

#sliding window approach
for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i,0])
    
X_test = np.array(X_test)

#reshaping test data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#performing prediction and reversing scaling so no longer between 0 and 1
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)


#root mean squared error
rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
print("RMSE: " + str(rmse))

#mean absolute percentage error
mape = np.mean(np.abs((Y_test - predictions) / Y_test)) * 100
print("MAPE: " + str(mape))

#plotting the data
train = data[:training_datarows]
valid = data[training_datarows:]
valid['Predictions'] = predictions

plt.figure(figsize=(13,5))
plt.title('CNN Stock Forecast')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Close Price USD ($)', fontsize=13)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Training data', 'Validation data', 'Prediction'], loc='upper left')
plt.show
