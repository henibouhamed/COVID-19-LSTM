# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 20:58:54 2020

@author: ASUS
"""

import numpy
from matplotlib import pyplot
from numpy import concatenate
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
dataset = read_csv('D:\\dataall19032020.txt', sep='\t', decimal=',',header=None, index_col=None)
values = dataset.values
values=values[:,2:]
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
X=scaled[:,0:11] 
Y=scaled[:,11]
r2=numpy.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],float)
#X=scaled[:,0:4] 
#Y=scaled[:,4]
from pandas import DataFrame
train = DataFrame()
val = DataFrame()
for j in range(10): 
  y_train, y_test, x_train, x_test = train_test_split(Y,X,test_size=0.10)
  x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
  x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
  # design network
  model = Sequential()
  model.add(LSTM(100,return_sequences=True,input_shape=(x_train.shape[1], x_train.shape[2])))
  model.add(Dropout(0.2))
  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(100,return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(100))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam')
  # fit network
  history = model.fit(x_train, y_train, epochs=1000, batch_size=1709, validation_data=(x_test, y_test), verbose=2, shuffle=False)
  train[str(j)] = history.history['loss']
  val[str(j)] = history.history['val_loss']
  # plot history
  # make a prediction
  yhat = model.predict(x_test)
  x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
#yhat = yhat.reshape((len(yhat), 1))
  inv_yhat = concatenate((yhat, x_test[:,1:],yhat),axis=1)
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]
  for i in range (190):
    if inv_yhat[i] < 0:
        inv_yhat[i]=inv_yhat[i]*-1
  # invert scaling for actual
  y_test = y_test.reshape((len(y_test), 1))
  inv_y = concatenate((y_test, x_test[:, 1:],y_test), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]
  from matplotlib import pyplot as plt
  plt.plot(inv_y, 'ro')
  plt.plot(inv_yhat, 'bo')
  plt.xlabel('Actual values')
  plt.ylabel('Predicted values')
  plt.show()
  name='D:\\books_read.png'+str(j)+'.jpeg'
  plt.savefig(name)
  # calculate RMSE
  
  for i in range(190):
    print('%.3f %.3f' % (inv_y[i],inv_yhat[i]))
  from sklearn.metrics import r2_score
  print(r2_score(inv_y, inv_yhat))
  r2[j]=r2_score(inv_y, inv_yhat)
#pyplot.plot(train, color='blue', label='train')
#pyplot.plot(val, color='orange' , label='validation')
#pyplot.title('model train vs validation loss')
#pyplot.ylabel('loss')
#pyplot.xlabel('epoch')
#pyplot.show()
#pyplot.savefig("D:\\allres.jpeg")
print(r2)