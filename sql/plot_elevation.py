#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot station elevation by geographical location.
"""


import mysql.connector
import getpass
import numpy as np
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------
# *** Connect to database

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
    
cursor = conn.cursor()


# -----------------------------------------------------------------------------
# ***  Access data

# Get number of stations with valid elevation:
sql = ('SELECT COUNT(*) FROM station_ids WHERE station_elevation IS NOT NULL '
       + 'AND country_code = "NO";')

# Execute SQL query:
cursor.execute(sql)
number_elevation = cursor.fetchall()[0][0]

# Column names (in SQL table and for dataframe):
columns = ['station_id', 'latitude', 'longitude', 'station_elevation']

# Select rows with valid station elevation.
sql = ('SELECT '
       + ', '.join(columns)
       + ' FROM station_ids WHERE (station_elevation IS NOT NULL '
       + 'AND country_code = "NO");')
cursor.execute(sql)

# Array for SQL data:
data = np.zeros((number_elevation, 4), dtype=np.float32)

# Fill dataframe with SQL rows:
idx_row = 0
for row in cursor.fetchall():
    for idx_column in range(4):
        data[idx_row, idx_column] = float(row[idx_column])
    idx_row += 1

# Create dataframe:
df = pd.DataFrame(data, columns=columns)

# Set dataframe column datatypes.
#dtypes = {'station_id': np.int16,
#           'latitude': np.float32,
#           'longitude': np.float32,
#           'station_elevation': np.float32}
#df.astype(dtypes)


# -----------------------------------------------------------------------------
# *** Plot elevation

sns.set()

cmap = sns.color_palette("BuGn_r")

cmap = sns.cubehelix_palette(rot=0.9,
                             gamma=5.0,
                             hue=1.0,
                             #light=0.2,
                             #dark=0.2,
                             as_cmap=True)

ax = sns.scatterplot(x='longitude',
                     y='latitude',
                     hue='station_elevation',
                     palette=cmap,
                     size=0.01,
                     legend=False,
                     data=df)

        
# -----------------------------------------------------------------------------
# *** Close objects

cursor.close()
conn.close()

