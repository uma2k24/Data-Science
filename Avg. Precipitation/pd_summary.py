import pandas as pd

totals = pd.read_csv('totals.csv').set_index(keys=['name']) # Read totals.csv into a Pandas dataframe
counts = pd.read_csv('counts.csv').set_index(keys=['name']) # Same for counts.csv

# 1: City with the lowest total precipitation over the year
totals_city = totals.sum(axis=1) #axis = 1 -> sum of rows
total_city_lowest = totals_city.idxmin(axis=0) #axis = 0 -> sum of columns
print('City with lowest total precipitation:\n', total_city_lowest)

# 2: Avg. precipitation in each month
totals_months = totals.sum(axis=0)
counts_sum = counts.sum(axis=0)
avg_precipitation_per_month = totals_months/counts_sum
print('Average precipitation in each month:\n', avg_precipitation_per_month)

# 3: Avg. precipitation in each city
counts_city = counts.sum(axis=1)
avg_precipitation_city = totals_city / counts_city
print('Average precipitation in each city:\n', avg_precipitation_city)