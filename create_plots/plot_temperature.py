#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot temperature by geographical location.
"""


import os
import mysql.connector
import getpass
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as geo
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# -----------------------------------------------------------------------------
# *** Parameters

# Plot anual mean of daily mean temperature in date range:
years = list(range(1900, 2018))

# Threshold number of valid observations per year, stations with fewer valid
# observations per year are excluded from plot:
num_obs_thr = 328

# Output path for plots:
path_plots = '/media/sf_D_DRIVE/Sonstiges/Weather_Data/plots'


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

sns.set()

# Using an INNER JOIN statement results in prohibitively slow performance.
# Instead, we create two temporary tables for the weather stations'
# geographical position, and the mean temperature in the specified time
# interval. The two temporary tables can then be joined on station ID.

# The temporal table for geographical location only needs to be created once.
sql = ('CREATE TEMPORARY TABLE station_ids_tmp '
       + 'SELECT '
       + '  station_ids.station_id, '
       + '  station_ids.longitude, '
       + '  station_ids.latitude '
       + 'FROM station_ids '
       + 'WHERE '
       + '  ((station_ids.longitude BETWEEN -15 AND 35) '
       + '   AND '
       + '  (station_ids.latitude BETWEEN 35 AND 75)); ')

# Execute SQL query:
cursor.execute(sql)

for year in years:

    print(('Year: ' + str(year)))

    # Create temporary table for stations with valid measurement:
    sql = ('CREATE TEMPORARY TABLE mean_temperature_tmp '
           + 'SELECT '
           + '  mean_temperature.station_id, '
           + '  AVG(mean_temperature.measurement) AS avg_temp, '
           + '  COUNT(mean_temperature.measurement) AS count_temp '
           + 'FROM mean_temperature '
           + 'WHERE '
           + '  ((mean_temperature.date BETWEEN \'{}-01-01\' AND \'{}-12-31\') '
           + '   AND '
           + '   (mean_temperature.quality = \'valid\')) '
           + 'GROUP BY mean_temperature.station_id;'
           ).format(year, year)

    # Execute SQL query:
    cursor.execute(sql)

    # Count number of valid observations for current time interval:
    sql = ('SELECT COUNT(*) '
           + 'FROM station_ids_tmp '
           + 'INNER JOIN mean_temperature_tmp  '
           + 'ON station_ids_tmp.station_id = mean_temperature_tmp.station_id; '
           )

    # Execute SQL query:
    cursor.execute(sql)
    number_observations = cursor.fetchall()[0][0]

    print(('Retrieving ' + str(number_observations) + ' measurements.'))

    # Column names (in SQL table and for dataframe):
    columns = ['station_id',
               'longitude',
               'latitude',
               'measurement',
               'observation_count']

    # Retrieve weather data:
    sql = ('SELECT '
           + '  station_ids_tmp.station_id, '
           + '  station_ids_tmp.longitude, '
           + '  station_ids_tmp.latitude, '
           + '  mean_temperature_tmp.avg_temp, '
           + '  mean_temperature_tmp.count_temp '
           + 'FROM station_ids_tmp '
           + 'INNER JOIN mean_temperature_tmp  '
           + 'ON station_ids_tmp.station_id = mean_temperature_tmp.station_id; '
           )

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

    # Number of stations before excluding stations with low number of valid
    # observations:
    sta_num_ttl = len(df_weather)

    # Exclude stations with low number of valid observations:
    df_weather = df_weather[np.greater(df_weather['observation_count'],
                                       num_obs_thr)]

    # Number of stations after thresholding:
    sta_num_thr = len(df_weather)

    print(('Excluded '
           + str(sta_num_ttl - sta_num_thr)
           + ' stations out of '
           + str(sta_num_ttl)
           + ' because there were less than '
           + str (num_obs_thr)
           + ' valid records for the year '
           + str(year)))

    # Drop temporary table pertaining to current time interval.
    sql = 'DROP TEMPORARY TABLE mean_temperature_tmp;'
    cursor.execute(sql)


    # -----------------------------------------------------------------------------
    # *** Create plot

    fgr = plt.figure(figsize=(5.6, 4.48), dpi=120)

    # Minimum and maximum temperatures, for scaling of colormap:
    temp_min = -100.0  # df_weather['measurement'].min()
    temp_max = 250.0  # df_weather['measurement'].max()

    # Prepare colour map:
    clr_norm = colors.Normalize(vmin=temp_min, vmax=temp_max)
    cmap = plt.cm.plasma  # plt.cm.winter

    # Get map with country borders as background:
    df_world = geo.read_file(geo.datasets.get_path('naturalearth_lowres'))

    # Plot country borders:
    ax = df_world.plot(figsize=(5.6, 4.48),
                       alpha=0.5,
                       facecolor='#dfdfdfff',
                       edgecolor='#707070ff',
                       ax=plt.gca())

    # Only plot Europe:
    ax.set_xlim(-15.0, 35.0)
    ax.set_ylim(35.0, 75.0)

    # Plot weather data:
    sns.scatterplot(x='longitude',
                    y='latitude',
                    hue='measurement',
                    palette=cmap,
                    size=0.01,
                    legend=False,
                    data=df_weather,
                    ax=ax)

    # Plot title:
    title = ('Mean annual temperature ' + str(year))
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Disable axis:
    plt.axis('off')

    # Tighter plot layout:
    try:
        plt.tight_layout(pad=0.5)
    except ValueError:
        pass

    # Save figure:
    fgr_path = os.path.join(path_plots,
                            ('mean_anual_temperature_' + str(year) + '.png')
                            )
    fgr.savefig(fgr_path)

    # Close figure:
    plt.close(fgr)

print('Done.')


# -----------------------------------------------------------------------------
# *** Close objects

cursor.close()
conn.close()
