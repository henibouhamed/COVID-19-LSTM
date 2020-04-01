# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:41:25 2020

@author: ASUS
"""
import numpy
from numpy import concatenate
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
datasetr = read_csv('D:\\recovered23032020.txt', sep='\t', decimal=',',header=None, index_col=None)
valuesr = datasetr.values
valuesr=valuesr[:,2:]
valuesr = valuesr.astype('float32')
# normalize features
scalerr = MinMaxScaler(feature_range=(0, 1))
scaledr = scalerr.fit_transform(valuesr)
X=scaledr[:,0:12] 
Y=scaledr[:,12]
rrecov2=numpy.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.],float)
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
  modelr = Sequential()
  modelr.add(LSTM(100,return_sequences=True,input_shape=(x_train.shape[1], x_train.shape[2])))
  modelr.add(Dropout(0.2))
  modelr.add(LSTM(100,return_sequences=True))
  modelr.add(Dropout(0.2))
  modelr.add(LSTM(100,return_sequences=True))
  modelr.add(Dropout(0.2))
  modelr.add(LSTM(100))
  modelr.add(Dense(1))
  modelr.compile(loss='mae', optimizer='adam')
# fit network
  historyr = modelr.fit(x_train, y_train, epochs=1000, batch_size=534, validation_data=(x_test, y_test), verbose=2, shuffle=False)
  train[str(j)] = historyr.history['loss']
  val[str(j)] = historyr.history['val_loss']
# plot history
# make a prediction
  yhat = modelr.predict(x_test)
  x_test = x_test.reshape((x_test.shape[0], x_test.shape[2]))
  #yhat = yhat.reshape((len(yhat), 1))
  inv_yhat = concatenate((x_test[:,:],yhat),axis=1)
  inv_yhat = scalerr.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,12]
  for i in range (60):
    if inv_yhat[i] < 0:
        inv_yhat[i]=inv_yhat[i]*-1
  # invert scaling for actual
  y_test = y_test.reshape((len(y_test), 1))
  inv_y = concatenate((x_test[:,:],y_test), axis=1)
  inv_y = scalerr.inverse_transform(inv_y)
  inv_y = inv_y[:,12]
  from matplotlib import pyplot as plt
  plt.plot(inv_y, 'ro')
  plt.plot(inv_yhat, 'bo')
  plt.xlabel('Actual values')
  plt.ylabel('Predicted values')
  #plt.show()
  name='D:\\recov.png'+str(j)+'.jpeg'
  plt.savefig(name)
  # calculate RMSE
  for i in range(60):
    print('%.3f %.3f' % (inv_y[i],inv_yhat[i]))
  from sklearn.metrics import r2_score
  print(r2_score(inv_y, inv_yhat))
  rrecov2[j]=r2_score(inv_y, inv_yhat)
#from matplotlib import pyplot
#pyplot.plot(train, color='blue', label='train')
#pyplot.plot(val, color='orange' , label='validation')
#pyplot.title('model train vs validation loss')
#pyplot.ylabel('loss')
#pyplot.xlabel('epoch')
#pyplot.show()
#pyplot.savefig("D:\\allresrecov.jpeg")
print(rrecov2)
