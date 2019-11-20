#!/bin/bash

# Set up new database for weather data. This script only sets up the database,
# use `create_tables.sh` to populate the database with tables.

# Set up database for user john (replace `password` with actual password):
echo 'Setting up new database as root user'
sudo mysql -u root -p
CREATE DATABASE  weather;
GRANT ALL ON weather.* TO 'john'@'localhost' IDENTIFIED BY 'password' WITH GRANT OPTION;
FLUSH PRIVILEGES;
exit;
