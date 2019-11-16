#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot temperature by geographical location.
"""


import mysql.connector
import getpass
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as geo
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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

print('Plot temperature by geographical location.')

print('Counting number of entries to retrieve from database.')

# Get number of stations with valid measurement:
sql = ('SELECT '
       + 'COUNT(*) '
       + 'FROM mean_temperature '
       + 'LEFT JOIN station_ids '
       + 'ON mean_temperature.station_id = station_ids.station_id '
       + 'WHERE '
       + '((mean_temperature.date = \'2006-05-10\') '
       + 'AND (mean_temperature.quality = \'valid\')) '
       + 'AND mean_temperature.station_id IN ('
       + 'SELECT station_ids.station_id FROM station_ids '
       + 'WHERE '
       + '((station_ids.longitude BETWEEN -15 AND 35) '
       + 'AND (station_ids.latitude BETWEEN 35 AND 75))'
       + ');')

# Execute SQL query:
cursor.execute(sql)
number_observations = cursor.fetchall()[0][0]

print(('Retrieving ' + str(number_observations) + ' measurements.'))

# Column names (in SQL table and for dataframe):
columns = ['station_id', 'longitude', 'latitude', 'measurement']

# Select rows from database.
sql = ('SELECT '
       + 'mean_temperature.station_id, station_ids.longitude, '
       + 'station_ids.latitude, mean_temperature.measurement '
       + 'FROM mean_temperature '
       + 'LEFT JOIN station_ids '
       + 'ON mean_temperature.station_id = station_ids.station_id '
       + 'WHERE '
       + '((mean_temperature.date = \'2006-05-10\') '
       + 'AND (mean_temperature.quality = \'valid\')) '
       + 'AND mean_temperature.station_id IN ('
       + 'SELECT station_ids.station_id FROM station_ids '
       + 'WHERE '
       + '((station_ids.longitude BETWEEN -15 AND 35) '
       + 'AND (station_ids.latitude BETWEEN 35 AND 75))'
       + ');')

cursor.execute(sql)

# Array for SQL data:
data = np.zeros((number_observations, len(columns)), dtype=np.float32)

# Fill dataframe with SQL rows:
idx_row = 0
for row in cursor.fetchall():
    for idx_column in range(len(columns)):
        data[idx_row, idx_column] = float(row[idx_column])
    idx_row += 1

# Create dataframe:
df_weather = pd.DataFrame(data, columns=columns)


# -----------------------------------------------------------------------------
# *** Create plot

sns.set()

fgr = plt.figure(figsize=(10, 8), dpi=120)

# Minimum and maximum temperatures, for scaling of colormap:
temp_min = df_weather['measurement'].min()
temp_max = df_weather['measurement'].max()

# Prepare colour map:
clr_norm = colors.Normalize(vmin=temp_min, vmax=temp_max)
cmap = plt.cm.plasma  # plt.cm.winter

# Get map with country borders as background:
df_world = geo.read_file(geo.datasets.get_path('naturalearth_lowres'))

# Plot country borders:
ax = df_world.plot(figsize=(10, 10),
                   alpha=0.5,
                   facecolor='#dfdfdfff',
                   edgecolor='#707070ff',
                   ax=plt.gca())

# Only plot Europe:
ax.set_xlim(-15.0,35.0)
ax.set_ylim(35.0,75.0)

# Plot weather data:
sns.scatterplot(x='longitude',
                y='latitude',
                hue='measurement',
                palette=cmap,
                size=0.01,
                legend=False,
                data=df_weather,
                ax=ax)

# Save figure:
fgr_path = '/home/john/Desktop/weather.png'
fgr.savefig(fgr_path)

# Close figure:
plt.close(fgr)
    

# -----------------------------------------------------------------------------
# *** Close objects

cursor.close()
conn.close()
