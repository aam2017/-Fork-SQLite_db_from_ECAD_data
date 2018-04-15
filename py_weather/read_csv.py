# -*- coding: utf-8 -*-

import csv
import datetime


def read_csv(strPath):
    """Load csv data to list."""
    # Open csv file:
    fleCsv = open(strPath, 'r')

    # Read csv file:
    csvIn = csv.reader(fleCsv,
                       delimiter='\n',
                       skipinitialspace=True)

    # Create empty list for CSV data:
    lstCsv = []

    # Loop through csv object to fill list with csv data:
    for lstTmp in csvIn:
        for strTmp in lstTmp:
            lstCsv.append(strTmp[:])

    # Close file:
    fleCsv.close()

    # Return list with csv data:
    return lstCsv


def delta_days(varDate01, varDate02):
    """Difference between two dates (YYYYMMDD) in days."""
    objDate01 = datetime.strptime(varDate01, "%Y%m%d")
    objDate02 = datetime.strptime(varDate02, "%Y%m%d")
    varDeltaDays = abs((objDate02 - objDate01).days)
    return varDeltaDays
