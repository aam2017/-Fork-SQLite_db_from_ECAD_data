#!/bin/bash

# Set up database for user john (replace `password` with actual password):
# echo 'Setting up new database as root user'
# sudo mysql -u root -p
# CREATE DATABASE  weather;
# GRANT ALL ON weather.* TO 'john'@'localhost' IDENTIFIED BY 'password' WITH GRANT OPTION;
# FLUSH PRIVILEGES;
# exit;

# Log in as non-root user:
echo 'Log in as non-root user'
mysql -u john -p

echo 'Creating tables'

# Parameters. One table will be created for each parameter.
parameters=(max_temperature,
            min_temperature,
            mean_temperature,
            precipitation,
            mean_pressure,
            cloud_cover,
            humidity,
            snow_depth,
            sunshine_duration,
            global_radiation,
            mean_wind_speed,
            max_wind_gust,
            wind_direction)

USE weather;

for param in ${parameters[@]}
do
  CREATE TABLE weather.${param}(
    auto_id INTEGER NOT NULL AUTO_INCREMENT,
    station_id INTEGER,
    source_id INTEGER,
    date DATE,
    measurement INTEGER,
    quality ENUM('valid', 'suspect', 'missing'),
    PRIMARY KEY (auto_id)
    )AUTO_INCREMENT=1;
done
