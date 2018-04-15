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
from utils import read_csv
from utils import delta_days


def load_data(lstFtr, strPthIn, strFleIn, varNumHdr, strPthHd, varStrtDate,
              varEndDate):
    """Load weather data from text files and save to hdf5 file."""
    # Number of weather features:
    varNumFtr = len(lstFtr)

    # Total number of days in time interval (plus one because the interval is
    # inclusive on both ends):
    varNumDays = delta_days(varEndDate, varStrtDate) + 1

    # File names of datafiles will be stored in a nested list, one list per
    # weather feature.
    lstFls = [None] * len(lstFtr)

    # Create hdf5 file:
    fleHd = h5py.File(strPthHd, 'w')

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

    # Create dataset within hdf5 file:
    objData = fleHd.create_dataset('weather_data',
                                   data=np.empty((varNumSta,
                                                  varNumFtr,
                                                  varNumDays)),
                                   dtype=np.int32)

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
                    # date, and not newer than specified end data:
                    if ((int(lstDataTmp[2]) >= varStrtDate)
                            and (int(lstDataTmp[2]) <= varEndDate)):

                        # Index of current date (i.e. days between start date
                        # and current date):
                        varIdxDay = delta_days(lstDataTmp[2], varStrtDate)

                        # Put measurement value into hdf5 object:
                        objData[idxSta, idxFtr, varIdxDay] = int(lstDataTmp[3])

    # Cloase hdf5 file:
    fleHd.close()
