#!/bin/bash

# Create tables in weather database. Use `create_database.sh` to set up the
# database first.

echo "Create tables for weather database."

echo "Please enter user name:"
read user_name

echo "Please enter password:"
read -s password

echo "Creating tables."

# Parameters. One table will be created for each parameter.
parameters=(max_temperature \
            min_temperature \
            mean_temperature \
            precipitation \
            mean_pressure \
            cloud_cover \
            humidity \
            snow_depth \
            sunshine_duratio \
            global_radiation \
            mean_wind_speed \
            max_wind_gust \
            wind_direction)

# Loop through parameters and create respective tables:
for param in ${parameters[@]}
do
  # SQL command:
  sql_cmd="CREATE TABLE weather.${param}(
    auto_id INTEGER NOT NULL AUTO_INCREMENT,
    station_id INTEGER,
    source_id INTEGER,
    date DATE,
    measurement INTEGER,
    quality ENUM('valid', 'suspect', 'missing'),
    PRIMARY KEY (auto_id)
    )AUTO_INCREMENT=1;"
  mysql -u ${user_name} -p${password} -D weather -e "${sql_cmd}"
  # echo "${sql_cmd}"
done

# Show resulting tables:
mysql -u ${user_name} -p${password} -D weather -e "SHOW TABLES;"
