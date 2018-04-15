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
import csv
import datetime
import numpy as np
import h5py
import pandas as pd
from utils import read_csv
from utils import delta_days


# -----------------------------------------------------------------------------
# *** Define parameters

# List of weather features to use (TG = mean temperature, PP = sea level
# pressure, CC = cloud cover in oktas):
#lstFtr = ['TG', 'PP', 'CC']
lstFtr = ['PP']

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
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# *** Prepare data

# Number of weather features:
varNumFtr = len(lstFtr)

# Check whether data has already been stored in hdf5 format:
if os.path.isfile(strPthHd):
    
    # TODO
    # Load file
    print('a')

else:

    # File names of datafiles will be stored in a nested list, one list per
    # weather feature.
    lstFls = [None] * len(lstFtr)

    # Create hdf5 file:
    fleHd = h5py.File(strPthHd, 'w')





datetime.timedelta()

    # Create dataset within hdf5 file:
    objData = fleHd.create_dataset('weather_data',
                                   dtype=np.float32)


    # ** Find complete datasets
    
    # Search for data text files - loop through features:
    for idxFtr in range(varNumFtr):

        # Get number of files (i.e. number of weather stations for which
        # record is available).

        # Appreviation for current feature:
        strFtr = lstFtr[idxFtr]

        # Complete path of current data directory:
        strPthTmp = strPthIn.format(strFtr)

        # List of files for current feature:
        lstFlsTmp = os.listdir(path=strPthTmp)

        # Only use file names corresponding to input file format:
        lstFlsTmp = [f.split('_')[1] for f in lstFlsTmp if \
                     strFleIn.format(strFtr, '').split('.')[0] in f]

        # List to set (lstFls now contains one set per weather feature, each
        # of these sets contains the file names for that feature, e.g.
        # 'STAID004923.txt', i.e. withouth prefix (i.e. without 'CC_'):
        lstFls[idxFtr] = set(lstFlsTmp)
        
    # ** Only select complete datasets
    
    # For now, only weather stations for which records for all available
    # features are available will be selected:
    lstFls = list(set.intersection(*lstFls))
    
    # Sort list of weather stations:
    lstFls = sorted(lstFls)
    
    # Number of datasets (weather stations):
    varNumSta = len(lstFls)

    # ** Load measurement data

    # Loop through stations:
    for idxSta in range(varNumSta):
        
        # Loop through features:
        for idxFtr in range(varNumFtr):

            # Path of current datafile:
            strPthTmp = (strPthIn.format(lstFtr[idxFtr])
                         + lstFtr[idxFtr]
                         + '_'
                         + lstFls[idxSta])

            # Load data from file:
            lstCsv = read_csv(strPthTmp)[varNumHdr:]

            # Loop line of input file:            
            for idxStr in range(len(lstCsv)):

                # Split each line:
                lstDataTmp = lstCsv[idxStr].split(',')

                # Each line contains the following five values:
                # - Station identifier
                # - Source identifier
                # - Date YYYYMMDD
                # - Measurement (e.g. temperature)
                # - Quality code (0='valid'; 1='suspect'; 9='missing')
                # If the data is valid (i.e. the quality code is 0), the date
                # and the measurement value are put into the array.
                if int(lstDataTmp[4]) == 0:

                    # Only import data if it is not older than specified start
                    # date:
                    if np.greater(int(lstDataTmp[2]), varStrtDate):

                    objData[]

                    int(lstDataTmp[2])
                    int(lstDataTmp[3])

varStrtDate

int(lstTmp[2])

lstCsv[0]

#aryCsv = np.array(lstCsv)


# Create dataset within hdf5 file:
#dtsDsgn = fleDsgn.create_dataset('design_matrix',
#                                 dtype=np.float32)

# ** Load data
    
# Cloase hdf5 file:
fleHd.close()


