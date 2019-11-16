# -*- coding: utf-8 -*-
"""
Write weather data into SQL database.
"""


import csv
import mysql.connector
import getpass
# import numpy as np
import os


# -----------------------------------------------------------------------------
# *** Parameters

# Path of station list csv file (parameter name, parameter abbreviation, and
# station ID left open):
# /media/sf_D_DRIVE/Sonstiges/Weather_Data/snow_depth/SD_STAID000001.txt
csv_path = '/media/sf_D_DRIVE/Sonstiges/Weather_Data/{}/{}_STAID{}.txt'

parameters = {'cloud_cover': 'CC',
              'global_radiation': 'QQ',
              'humidity': 'HU',
              'max_temperature': 'TX',
              'max_wind_gust': 'FX',
              'mean_pressure': 'PP',
              'mean_temperature': 'TG',
              'mean_wind_speed': 'FG',
              'min_temperature': 'TN',
              'precipitation': 'RR',
              'snow_depth': 'SD',
              'sunshine_duration': 'SS',
              'wind_direction': 'DD'}


# -----------------------------------------------------------------------------
# *** Connect to database

print('Write weather data into SQL database.')

# Prompt for user name an password.
user_name = input('Please enter user name: ')
password = getpass.getpass('Please enter password: ')

conn = mysql.connector.connect(host='localhost',
                               database='weather',
                               user=user_name,
                               password=password)

if conn.is_connected():
    print('Connected to MySQL database')
else:
    assert False, 'Connection failed.'


# -----------------------------------------------------------------------------
# *** Write data from disk to database

# SQL query for adding rows. Parameter name and five values left open.
query = ('INSERT INTO weather.{} ('
         + 'station_id, '
         + 'source_id, '
         + 'date, '
         + 'measurement, '
         + 'quality'
         + ') VALUES (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\');')

cursor = conn.cursor()

for key, value in parameters.items():

    count = 0

    print('Writing data for parameter: ' + key)

    # Directory containing csv files of current parameter:
    param_path = csv_path.format(key, value, '')
    param_path = os.path.split(param_path)[0]

    # Get list of csv files with weather data:
    file_list = os.listdir(param_path)

    # There may be files other than csv files with weather data in the
    # directory. Only attempt to read files with the following pattern in their
    # file name:
    pattern = (value + '_STAID')
    for file_name in file_list:
        if pattern in file_name:
            # Complete file path:
            file_path = os.path.join(param_path, file_name)

            # Open csv file
            file = open(file_path, 'r')
            text = csv.reader(file,
                              delimiter=',',
                              skipinitialspace=True)

            # We need to skip the header of the csv file. The line before the
            # start of the record starts with the string 'STAID'. We ignore
            # lines until this string is matched.
            switch = True

            # Loop through lines of csv file:
            for lst_lne in text:

                # Skip csv file header.
                if switch:
                    # Avoid empty lines:
                    if len(lst_lne) >= 5:
                        if lst_lne[0] == 'STAID':
                            switch = False

                else:
                    # Read weather record.
                    station_id = lst_lne[0]
                    source_id = lst_lne[1]
                    year = lst_lne[2][0:4]
                    month = lst_lne[2][4:6]
                    day = lst_lne[2][6:8]
                    date = (year + '-' + month + '-' + day)
                    measurement = lst_lne[3]

                    # Quality code (0='valid'; 1='suspect'; 9='missing').
                    if lst_lne[4] == '0':
                        quality = 'valid'
                    elif lst_lne[4] == '1':
                        quality = 'suspect'
                    elif lst_lne[4] == '9':
                        quality = 'missing'
                    else:
                        warning = ('Warning: Unknown quality code in input '
                                   + 'data, should be `0` for `valid`, `1` for'
                                   + ' `suspect`, `9` for `missing`. Will '
                                   + 'label respective record as `suspect`.')
                        print(warning)
                        quality = 'suspect'

                    # Complete SQL query:
                    sql = query.format(key,
                                       station_id,
                                       source_id,
                                       date,
                                       measurement,
                                       quality)

                    # Execute SQL query:
                    cursor.execute(sql)
                    count += 1

                    # Commit transaction:
                    if (count % 1000000) == 0:
                        print('Committing transaction: ' + str(count))
                        conn.commit()

        else:
            print('Skipping file: ' + file_name)

    # Final commit:
    conn.commit()

    print('Committed ' + str(count) + ' measurements for parameter ' + key)

print('Done.')


# -----------------------------------------------------------------------------
# *** Close objects

file.close()
cursor.close()
conn.close()
