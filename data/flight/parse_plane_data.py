# Copyright Nicolo' Fusi and James Hensman 2013
# Modified 23/06/2015: Jun Wei Ng
# Saves as numpy array intead of pandas dataframe
import numpy as np
import pandas

data = pandas.read_csv('2008.csv')
data = data[data['Cancelled'] == 0]
data = data[data['Diverted'] == 0]
plane_data  = pandas.read_csv('plane-data.csv', na_values='None')
merged = pandas.merge(data, plane_data, left_on='TailNum', right_on='tailnum')
merged = merged.rename(columns={'year': 'plane_age'})
filtered = merged[['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'ArrTime', 'ArrDelay', 'AirTime', 'Distance', 'plane_age']]
# Filter out plane deployment years == 0
filtered = filtered[filtered['plane_age'] != 0]
# Transform plane deployment year in plane age
filtered['plane_age'] = filtered['Year'] - filtered['plane_age']
filtered = filtered.dropna()
#filtered.save('filtered_data.pickle')
np.save('plane_data.npy',filtered.values)
