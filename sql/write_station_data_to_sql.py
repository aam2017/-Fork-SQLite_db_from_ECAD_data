# -*- coding: utf-8 -*-
"""
Read weather station metadata (name, position, etc.) into SQL database.
"""


import csv
import mysql.connector
import getpass
import numpy as np
from utils import dms_to_dec


# -----------------------------------------------------------------------------
# *** Parameters

# Path of station list csv file:
csv_path = '/media/sf_D_DRIVE/Sonstiges/Weather_Data/ECA_blend_station_tg.txt'

# Number of header lines to skip:
lines_skip = 18


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


# -----------------------------------------------------------------------------
# *** Write data from disk to database

# Open csv file
file = open(csv_path, 'r')

text = csv.reader(file,
                  delimiter=',',
                  skipinitialspace=True)

# SQL query for adding rows:
query = ('INSERT INTO weather.station_ids ('
         + 'station_id, '
         + 'station_name, '
         + 'country_code, '
         + 'latitude, '
         + 'longitude, '
         + 'station_elevation'
         + ') VALUES (\'{}\', \'{}\', \'{}\', \'{}\', \'{}\', \'{}\')')

cursor = conn.cursor()

# Loop through csv object and fill database
print('Writing text to database.')
count = 0
for lst_lne in text:
    if count > lines_skip:
        # Print initial lines for QA:
        if count < 30:
            print(lst_lne)
        # Get parameters from csv string:
        station_id = int(lst_lne[0])
        # Replace quotations marks from station name.
        station_name = lst_lne[1].strip().replace('\'', ' ').replace('\"', ' ')
        country_code = lst_lne[2].strip()
        latitude = np.around(dms_to_dec(lst_lne[3]), decimals=8)
        longitude = np.around(dms_to_dec(lst_lne[4]), decimals=8)
        station_elevation = int(lst_lne[5])
        # Complete SQL query:
        sql = query.format(station_id,
                           station_name,
                           country_code,
                           latitude,
                           longitude,
                           station_elevation)
        # Execute SQL query:
        cursor.execute(sql)
    count += 1
    # Commit transaction:
    if (count % 10000) == 0:
        print('Committing transaction. Number of lines: ' + str(count))
        conn.commit()
# Final commit:
conn.commit()

print('Done.')

# Replace invalid station elevation values (-999 or -9999) by NULL:
sql_null = ('UPDATE station_ids SET station_elevation = NULL '
            + 'WHERE station_elevation LIKE "%999%";')
cursor.execute(sql_null)
conn.commit()


# -----------------------------------------------------------------------------
# *** Close objects
        
file.close()
cursor.close()
conn.close()
