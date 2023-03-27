import os
from datetime import datetime, timedelta
import pandas as pd
# scrape hourly data
dirs = [
    r'C:\Users\Patrick\OneDrive - The University of Western Ontario\Documents\Research\MITACS\Code\src\main\resources\data\ontario\demand\actual',
    r'C:\Users\Patrick\OneDrive - The University of Western Ontario\Documents\Research\MITACS\Code\src\main\resources\data\ontario\demand\forecast',
    r'C:\Users\Patrick\OneDrive - The University of Western Ontario\Documents\Research\MITACS\Code\src\main\resources\data\ontario\pricing',
    r'C:\Users\Patrick\OneDrive - The University of Western Ontario\Documents\Research\MITACS\Code\src\main\resources\data\ontario\supply'
]

for directory in dirs:
    os.chdir(directory)
    exec(open('update_ieso_data.py').read())

os.chdir(r'C:\Users\Patrick\OneDrive - The University of Western Ontario\Documents\Research\MITACS\Code\src\main\resources\data\ontario')

supply_file = './supply/csv_files/ieso_supply.csv'
demand_file = './demand/actual/ieso_demand.csv'
price_file = './pricing/ieso_price.csv'


def convert_time(time):
    try:
        return datetime.strptime(time, '%Y-%m-%d %H')
    except ValueError:
        return datetime.strptime(time.replace('24', '23'), '%Y-%m-%d %H') + timedelta(hours=1)


# combined hourly data on intersection of observation times and write to csv
print('Combining and writing data...')
supply = pd.read_csv(supply_file)
demand = pd.read_csv(demand_file)
price = pd.read_csv(price_file)

supply['Time'] = supply.apply(lambda x: convert_time(x['Date'] + ' ' + str(x['Hour'])), axis=1)
demand['Time'] = supply.apply(lambda x: convert_time(x['Date'] + ' ' + str(x['Hour'])), axis=1)
price['Time'] = supply.apply(lambda x: convert_time(x['Date'] + ' ' + str(x['Hour'])), axis=1)

supply.drop(['Date', 'Hour'], inplace=True, axis=1)
demand.drop(['Date', 'Hour'], inplace=True, axis=1)
price.drop(['Date', 'Hour'], inplace=True, axis=1)

ieso_data = pd.merge(supply, demand, on='Time', copy=False)
ieso_data = pd.merge(ieso_data, price, on='Time', copy=False)

ieso_data.to_csv('ieso_combined.csv')
