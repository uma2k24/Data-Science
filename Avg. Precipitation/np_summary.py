import numpy as np

data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

# 1: City with the lowest total precipitation over the year
precipitation_sum = np.sum(totals, axis=1) #axis = 1 -> sum of rows
lowest_precipitation = np.argmin(precipitation_sum)
print('Row with lowest total precipitation:')
print(lowest_precipitation)

# 2: Avg. precipitation in each month
total_precipitation_per_month = np.sum(totals, axis=0) #axis = 0 -> sum of columns
months = np.sum(counts, axis=0)
avg_precipitation_per_month = total_precipitation_per_month/months
print('Average precipitation in each month:')
print(avg_precipitation_per_month)

# 3: Avg. precipitation in each city
total_precipitation_city = np.sum(totals, axis=1)
cities = np.sum(counts, axis=1)
avg_precipitation_city = total_precipitation_city/cities
print('Average precipitation in each city:')
print(avg_precipitation_city)

# 4: Total precipitation for each quarter in each city
n = len(totals)
quarter_total = np.reshape(totals, (4*n, 3))
quarter_precipitation_city = np.sum(quarter_total, axis=1)
quarter_precipitation_city.shape = (n, 4)
print('Quarterly precipitation totals:')
print(quarter_precipitation_city)