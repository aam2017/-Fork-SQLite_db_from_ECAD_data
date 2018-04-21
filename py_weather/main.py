# -*- coding: utf-8 -*-
"""Tensorflow weather prediction."""

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
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# *** Define parameters

# List of weather features to use (TG = mean temperature, PP = sea level
# pressure, CC = cloud cover in oktas, RR = precipitation amount in 0.1 mm,
# DD = wind direction, HU = humidity):
# lstFtr = ['CC', 'PP', 'RR', 'TG', 'TN', 'TX', 'DD', 'HU']
lstFtr = ['CC', 'PP', 'RR', 'TG', 'TN', 'TX']

# Input data path of text files (weather feature left open):
strPthIn = '/media/sf_D_DRIVE/eu_weather/ECA_blend_{}/'

# File name of input text files (weather feature & station ID left open):
strFleIn = '{}_STAID{}.txt'

# Number of header lines in data text files:
varNumHdr = 21

# Line containing station name (in csv header):
varHdrLneSta = 12

# Path of hdf5 file:
# strPthHd = '/media/sf_D_DRIVE/eu_weather/hdf5/data.hdf5'
strPthHd = '/home/john/Documents/data.hdf5'

# Path of station list:
strPthLst = '/media/sf_D_DRIVE/eu_weather/hdf5/station_list.txt'

# Load data starting from this data (YYYYMMMDD):
varStrtDate = 19800101

# Load data until this date (YYYMMMDD):
varEndDate = 20180331

# Predict weather based on past x days:
varNumPast = 5

# Predict weather for xth day into the future (i.e. 1 for following day):
varNumPre = 1

# Station ID for which to predict:
varBase = 9

# Which feature to predict (index in list, see above):
# varFtrPrd = 2

# Use last x days as test data set (the preceeding time period is used as
# taining data set).
varDaysTest = 365

# Number of epochs:
varEpch = 100

# Output folder for plots:
# strPthPlot = '/Users/john/Dropbox/Sonstiges/Weather_Plots/'
strPthPlot = '/home/john/Dropbox/Sonstiges/Weather_Plots/'

# Figure dpi:
varDpi = 96

# Figure size:
varSizeX = 2200
varSizeY = 400
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
    load_data(lstFtr, strPthIn, strFleIn, varNumHdr, varHdrLneSta, strPthHd,
              strPthLst, varStrtDate, varEndDate)

    # Load weather data from hdf5 file:
    fleHd = h5py.File(strPthHd, 'r')
    objData = fleHd['weather_data']

# Data into memory:
aryData = np.array(objData, dtype=np.float32)

# Number of datasets (weather stations):
varNumSta = aryData.shape[0]

# Number of weather features:
varNumFtr = aryData.shape[1]

# Number of days for which data is available:
varNumDays = aryData.shape[2]

# Invalid values coded as:
varInvld = -9999.0

# Replace invalid values by median value for respective feature:
for idxFtr in range(varNumFtr):
    varTmpFtrMne = np.median(aryData[:, idxFtr, :])
    aryLgcInvl = np.equal(aryData[:, idxFtr, :], varInvld)
    aryDataTmp = aryData[:, idxFtr, :]
    aryDataTmp[aryLgcInvl] = varTmpFtrMne
    aryData[:, idxFtr, :] = aryDataTmp
    # print(np.sum(aryLgcInvl))

# Training data set - use this data for training (i.e. all features from all
# stations over the training period):
aryTrain = np.array(aryData[:, :, 0:(-(varDaysTest + 10))],
                    dtype=np.float32)

# Test data set - use this data for prediction during test (i.e. all features
# from all stations over the test preiod, e.g. last year):
aryTest = np.array(aryData[:, :, -varDaysTest:],
                   dtype=np.float32)

# Normalise the training data to range 0 to 1 (separately for each feature):
for idxFtr in range(varNumFtr):
    # vecTmp = aryTrain[:, idxFtr, :]
    # varTmpMin = np.min(vecTmp)
    # vecTmp = np.subtract(vecTmp, varTmpMin)
    # varTmpMax = np.max(vecTmp)
    # vecTmp = np.divide(vecTmp, varTmpMax)
    # aryTrain[:, idxFtr, :] = vecTmp
    vecTmp = aryTrain[:, idxFtr, :]
    varTmpMne = np.mean(vecTmp)
    vecTmp = np.subtract(vecTmp, varTmpMne)
    varTmpSd = np.std(vecTmp)
    vecTmp = np.divide(vecTmp, varTmpSd)
    # Remove extreme cases:
    vecLgc = np.greater(np.absolute(vecTmp), 2.5)
    vecTmp[vecLgc] = 0.0
    aryTrain[:, idxFtr, :] = vecTmp

# Normalise the test data to range 0 to 1 (separately for each feature):
for idxFtr in range(varNumFtr):
    # vecTmp = aryTest[:, idxFtr, :]
    # varTmpMin = np.min(vecTmp)
    # vecTmp = np.subtract(vecTmp, varTmpMin)
    # varTmpMax = np.max(vecTmp)
    # vecTmp = np.divide(vecTmp, varTmpMax)
    # aryTest[:, idxFtr, :] = vecTmp
    vecTmp = aryTest[:, idxFtr, :]
    varTmpMne = np.mean(vecTmp)
    vecTmp = np.subtract(vecTmp, varTmpMne)
    varTmpSd = np.std(vecTmp)
    vecTmp = np.divide(vecTmp, varTmpSd)
    aryTest[:, idxFtr, :] = vecTmp

# Training data set - predict this data during training (i.e. all features from
# one station over the training period):
# vecTrainPrd = aryTrain[varBase, varFtrPrd, :]
aryTrainPrd = aryTrain[varBase, :, :]

# Test data set - predict this data during test (i.e. all features from one
# station over the test period, e.g. last year):
aryTestPrd = aryTest[varBase, :, :]

# Reshape training data to aryTrain[days, (station * feature)]
aryTrain = np.reshape(aryTrain,
                      ((aryTrain.shape[0] * aryTrain.shape[1]),
                       aryTrain.shape[2])
                      ).T

# Reshape test data to aryTest[days, (station * feature)]
aryTest = np.reshape(aryTest,
                     ((aryTest.shape[0] * aryTest.shape[1]),
                      aryTest.shape[2])
                     ).T

del(aryData)
del(aryDataTmp)
del(aryLgcInvl)
del(vecTmp)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Plot data to be predicted

# Create figure:
fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                            (varSizeY * 0.5) / varDpi),
                   dpi=varDpi)

# Loop through features (separate subplots):
for idxFtr in range(varNumFtr):

    # Create axis:
    axs01 = fgr01.add_subplot(1, varNumFtr, (idxFtr + 1))

    # Plot depth profile for current input file:
    plt01 = axs01.plot(aryTestPrd[idxFtr, :])

# Output file name:
strPltTmp = (strPthPlot + 'data_to_be_predicted.png')

# Save figure:
fgr01.savefig(strPltTmp,
              facecolor='w',
              edgecolor='w',
              orientation='landscape',
              transparent=False,
              frameon=None)

# Close figure:
plt.close(fgr01)
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Train model

# Placeholder
# objX = tf.placeholder(dtype=tf.float32,
#                       shape=[varNumPast, (varNumSta * varNumFtr)])
objX = tf.placeholder(dtype=tf.float32,
                      shape=[1, (varNumSta * varNumFtr * varNumPast)])
objY = tf.placeholder(dtype=tf.float32, shape=[varNumFtr, 1])


# Initializers
varSigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg",
                                                     distribution="uniform",
                                                     scale=varSigma)
bias_initializer = tf.zeros_initializer()

# Model architecture parameters
n_input = (varNumSta * varNumFtr * varNumPast)
n_neurons_1 = 8192
n_neurons_2 = 4096
n_neurons_3 = 2048
n_neurons_4 = 1024
n_neurons_5 = 512
n_neurons_6 = 256
n_neurons_7 = 128
# n_target = 1
n_target = varNumFtr

print('n_input')
print(n_input)

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_input, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Layer 5: Variables for hidden weights and biases
W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))

# Layer 6: Variables for hidden weights and biases
W_hidden_6 = tf.Variable(weight_initializer([n_neurons_5, n_neurons_6]))
bias_hidden_6 = tf.Variable(bias_initializer([n_neurons_6]))

# Layer 7: Variables for hidden weights and biases
W_hidden_7 = tf.Variable(weight_initializer([n_neurons_6, n_neurons_7]))
bias_hidden_7 = tf.Variable(bias_initializer([n_neurons_7]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_7, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))
# bias_out = tf.Variable(bias_initializer([1, varNumFtr]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(objX, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))
hidden_6 = tf.nn.relu(tf.add(tf.matmul(hidden_5, W_hidden_6), bias_hidden_6))
hidden_7 = tf.nn.relu(tf.add(tf.matmul(hidden_6, W_hidden_7), bias_hidden_7))

# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_7, W_out), bias_out))

print('out.shape')
print(out.shape)

print('objY.shape')
print(objY.shape)

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, objY))

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(mse)

# Make Session
net = tf.Session()

# Run initializer
net.run(tf.global_variables_initializer())

# Index of last day in training data set from which to make predictions (total
# number of days in training data set minus lenght of prediction period minus
# days in the future to predict):
varLstDay = (aryTrain.shape[1] - varNumPast - varNumPre)

# Indices of days from which to make prediction:
vecIdx = np.arange(0, varLstDay, dtype=np.int32)

for idxEpch in range(varEpch):

    print(('--Epoch: ' + str(idxEpch)))

    # Shuffle training data indices:
    vecIdxPerm = np.random.permutation(vecIdx)

    # Loop through days:
    for idxDay in range(vecIdxPerm.shape[0]):

        # Data used for prediction:
        aryTmpX = aryTrain[idxDay:(idxDay + varNumPast), :]

        aryTmpX = aryTmpX.reshape(1, (aryTmpX.shape[0] * aryTmpX.shape[1]))

        # Data point to predict:
        varTmpY = np.array(aryTrainPrd[:, (idxDay + varNumPast + varNumPre)],
                           ndmin=2).T

        # Run optimizer with batch
        net.run(opt, feed_dict={objX: aryTmpX, objY: varTmpY})

        # Show progress
        if np.mod(idxDay, 10000) == 0:

            # Number of days to predict from in test data set:
            varNumTstDays = (aryTest.shape[0] - varNumPast - varNumPre)

            # Array for test predictions:
            aryPrdTst = np.zeros((varNumTstDays, varNumFtr),
                                 dtype=np.float32)

            # Loop through days in test data set:
            for idxDayTst in range(varNumTstDays):

                # aryTmpXtst = aryTest[idxDayTst:varNumPast, :]

                # print(idxDayTst)

                aryTestTmp = aryTest[idxDayTst:(idxDayTst + varNumPast), :]

                aryTestTmp = aryTestTmp.reshape(1,
                                                (aryTestTmp.shape[0]
                                                 * aryTestTmp.shape[1])
                                                )

                # Prediction
                pred = net.run(out, feed_dict={objX: aryTestTmp})

                # print('type(pred)')
                # print(type(pred))

                # print('aryPrdTst.shape')
                # print(aryPrdTst.shape)

                # print('pred.shape')
                # print(pred.shape)

                # print('pred')
                # print(pred)

                # `pred` has as many elements as there are features - why? Are
                # all features predicted?

                aryPrdTst[idxDayTst, :] = np.copy(pred[:, 0])

            # Error:
            # print('aryTestPrd.shape')
            # print(aryTestPrd.shape)
            # print('aryPrdTst.shape')
            # print(aryPrdTst.shape)
            aryErr = np.subtract(aryTestPrd[:, (varNumPast + varNumPre):],
                                 aryPrdTst.T)

            # Create figure:
            fgr01 = plt.figure(figsize=((varSizeX * 0.5) / varDpi,
                                        (varSizeY * 0.5) / varDpi),
                               dpi=varDpi)

            # Loop through features (separate subplots):
            for idxFtr in range(varNumFtr):

                # Create axis:
                axs01 = fgr01.add_subplot(1, varNumFtr, (idxFtr + 1))

                # Plot depth profile for current input file:
                plt01 = axs01.plot(aryErr[idxFtr, :])

            # Output file name:
            strPltTmp = (strPthPlot
                         + 'prediction_epoch_'
                         + str(idxEpch)
                         + '_day_'
                         + str(idxDay)
                         + '.png')

            # Save figure:
            fgr01.savefig(strPltTmp,
                          facecolor='w',
                          edgecolor='w',
                          orientation='landscape',
                          transparent=False,
                          frameon=None)

            # Close figure:
            plt.close(fgr01)

# Print final MSE after Training
# mse_final = net.run(mse, feed_dict={objX: vecTest, objY: y_test})
# print(mse_final)

# -----------------------------------------------------------------------------
