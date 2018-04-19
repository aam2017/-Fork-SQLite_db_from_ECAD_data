#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2018  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

import os
# import csv
# import datetime
import numpy as np
import h5py
# import pandas as pd
from load_data import load_data
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
# from sklearn.preprocessing import MinMaxScaler


# -----------------------------------------------------------------------------
# *** Define parameters

# List of weather features to use (TG = mean temperature, PP = sea level
# pressure, CC = cloud cover in oktas):
lstFtr = ['CC', 'PP', 'RR', 'TG', 'TN', 'TX']

# Input data path of text files (weather feature left open):
strPthIn = '/media/sf_D_DRIVE/eu_weather/ECA_blend_{}/'

# File name of input text files (weather feature & station ID left open):
strFleIn = '{}_STAID{}.txt'

# Number of header lines in data text files:
varNumHdr = 21

# Path of hdf5 file:
strPthHd = '/media/sf_D_DRIVE/eu_weather/hdf5/data.hdf5'

# Load data starting from this data (YYYYMMMDD):
varStrtDate = 19800101

# Data until this date (YYYMMMDD):
varEndDate = 20180331

# Predict weather based on past x days:
varNumPast = 5

# Predict weather for next x days:
varNumPre = 1

# Station ID for which to predict:
varBase = 432
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Preparations

# Check whether data has already been stored in hdf5 format:
if os.path.isfile(strPthHd):

    # Load weather data from hdf5 file:
    fleHd = h5py.File(strPthHd, 'r')
    objData = fleHd['weather_data']

else:

    # Load weather data from text files, and save to hdf5 file:
    load_data(lstFtr, strPthIn, strFleIn, varNumHdr, strPthHd, varStrtDate,
              varEndDate)

    # Load weather data from hdf5 file:
    fleHd = h5py.File(strPthHd, 'r')
    objData = fleHd['weather_data']

# Number of datasets (weather stations):
varNumSta = objData.shape[0]

# Number of weather features:
varNumFtr = objData.shape[1]

# Number of days for which data is available:
varNumDays = objData.shape[2]
# -----------------------------------------------------------------------------


## -----------------------------------------------------------------------------
#
#
#
## The data have the following shape: objData[station, weather-feature, day]
#aryData = np.array(objData)
#
#vecBase = aryData[varBase, 0, :]
#
#aryData = np.reshape(aryData,
#                     ((aryData.shape[0] * aryData.shape[1]),
#                      aryData.shape[2])
#                     )
#
#
## Training and test data
#train_start = 0
#train_end = (varNumDays - (2 * 365))
#test_start = train_end
#test_end = varNumDays
#data_train = aryData[:, np.arange(train_start, train_end)]
#data_test = aryData[:, np.arange(test_start, test_end)]
#
#
#
#
#
## Scale data
## ...
#
#
## Placeholder
#X = tf.placeholder(dtype=tf.float32, shape=[(varNumSta * varNumFtr), varNumPast])
#Y = tf.placeholder(dtype=tf.float32, shape=[varNumFtr, varNumPre])
#
#
## Initializers
#sigma = 1
#weight_initializer = tf.variance_scaling_initializer(mode="fan_avg",
#                                                     distribution="uniform",
#                                                     scale=sigma)
#bias_initializer = tf.zeros_initializer()
#
#
## Model architecture parameters
#n_stocks = (varNumSta * varNumFtr)
#n_neurons_1 = 1024
#n_neurons_2 = 512
#n_neurons_3 = 256
#n_neurons_4 = 128
#n_target = 1
## Layer 1: Variables for hidden weights and biases
#W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
#bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
## Layer 2: Variables for hidden weights and biases
#W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
#bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
## Layer 3: Variables for hidden weights and biases
#W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
#bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
## Layer 4: Variables for hidden weights and biases
#W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
#bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
#
## Output layer: Variables for output weights and biases
#W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
#bias_out = tf.Variable(bias_initializer([n_target]))
#
#
## Hidden layer
#hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
#hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
#hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
#hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
#
## Output layer (must be transposed)
#out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
#
## Cost function
#mse = tf.reduce_mean(tf.squared_difference(out, Y))
#
## Optimizer
#opt = tf.train.AdamOptimizer().minimize(mse)
#
## Make Session
#net = tf.Session()
## Run initializer
#net.run(tf.global_variables_initializer())
#
#
#
## Make Session
#net = tf.Session()
## Run initializer
#net.run(tf.global_variables_initializer())
#
## Setup interactive plot
##plt.ion()
##fig = plt.figure()
##ax1 = fig.add_subplot(111)
##line1, = ax1.plot(y_test)
##line2, = ax1.plot(y_test*0.5)
##plt.show()
#
## Number of epochs and batch size
#epochs = 10
#batch_size = 256
#
#shuffle_indices = np.random.permutation(np.arange(varNumDays))
#
#for e in range(epochs):
#
#    # Shuffle training data
#    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
#    X_train = X_train[shuffle_indices]
#    y_train = y_train[shuffle_indices]
#
#    # Minibatch training
#    for i in range(0, len(y_train) // batch_size):
#        start = i * batch_size
#        batch_x = X_train[start:start + batch_size]
#        batch_y = y_train[start:start + batch_size]
#        # Run optimizer with batch
#        net.run(opt, feed_dict={X: batch_x, Y: batch_y})
#
#        # Show progress
#        if np.mod(i, 5) == 0:
#            # Prediction
#            pred = net.run(out, feed_dict={X: X_test})
#            line2.set_ydata(pred)
#            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
#            file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.jpg'
#            plt.savefig(file_name)
#            plt.pause(0.01)
## Print final MSE after Training
#mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
#print(mse_final)
#
## -----------------------------------------------------------------------------
#
#
#
