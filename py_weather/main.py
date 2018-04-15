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


# -----------------------------------------------------------------------------
# *** Define parameters

# List of weather features to use (TG = mean temperature, PP = sea level
# pressure, CC = cloud cover in oktas):
lstFtr = ['TG', 'PP', 'CC']

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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------

# The data have the following shape: objData[station, weather-feature, day]

# Number of datasets (weather stations):
varNumSta = objData.shape[0]

# Number of weather features:
varNumFtr = objData.shape[1]



# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[varNumSta, varNumFtr, varNumPast])
Y = tf.placeholder(dtype=tf.float32, shape=[varNumFtr, varNumPre])

# -----------------------------------------------------------------------------

print('objData.shape')
print(objData.shape)

aaa = objData[520:540, 0, -1000:].T
