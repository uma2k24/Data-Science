# Utsav Anantbhat


import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def data_in(stations_file, cities_file):

    stations = pd.read_json(stations_file, lines=True) #read the files
    cities = pd.read_csv(cities_file)

    stations['avg_tmax'] = stations['avg_tmax']/10 #divide by 10 since column in the weather data is °C×10 

    #cities.dropna(subset=['area', 'population'], inplace=True) #remove cities missing either their area or population; reference: https://www.quicktable.io/apps/how-to-remove-rows-where-specified-column-are-blank-in-csv/
    cities = cities[~cities['area'].isnull() & ~cities['population'].isnull()]
    cities['area'] = cities['area']/(10**6) #area converted to km^2 from m^2
    cities['pop_density'] = cities['population']/(cities['area']) #calculate population density 


    cities = cities[cities['area'] <= 10000] #exclude cities with area greater than 10000 km^2

    return cities, stations


# Implement the haversine function to calculate distance between one city and every station; 
# sources/references: https://devpress.csdn.net/python/62fdab7fc677032930804307.html, 
# https://stackoverflow.com/questions/40452759/pandas-latitude-longitude-to-distance-between-successive-rows,
# https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas/29546836#29546836
def distance(city, stations):

    """
    city_lat = city['latitude'] 
    city_lon = city['longitude']
    station_lat = stations['latitude']
    station_lon = stations['longitude']
    city_lat, city_lon, station_lat, station_lon = map(np.radians, [city_lat, city_lon, station_lat, station_lon]) #convert angles from degrees to radians for the 4 values
    """

    stations_lat = np.deg2rad(stations['latitude']) #converting values from degrees to radians using numpy deg2rad
    city_lat = np.deg2rad(city['latitude'])
    stations_lon = np.deg2rad(stations['longitude'])
    city_lon = np.deg2rad(city['longitude'])

    diff_lat = stations_lat - city_lat
    diff_lon = stations_lon - city_lon #latitude and longitude difference: station - city
    hsine = np.sin(diff_lat / 2) ** 2 + np.cos(city['latitude']) * np.cos(stations['latitude']) * np.sin(diff_lon / 2) ** 2 #haversine formula
    c = 2 * np.arcsin(np.sqrt(hsine))
    meters = 6371000 * c #radius of Earth in meters: 6,371,000
    return meters


# Entity Resolution
# Implement funtion that returns the best value you can find for 'avg_tmax' for that one city, from the list of all weather stations
def best_tmax(city, stations):
    return stations.iloc[np.argmin(distance(city, stations))].avg_tmax


# Main function
def main():

    station_arg = sys.argv[1] # Take the file as inputs from the command line (system arguments)
    cities_arg = sys.argv[2]
    output_arg = sys.argv[3]

    cities, stations = data_in(station_arg, cities_arg)
    cities['tmax'] = cities.apply(best_tmax, stations=stations, axis = 1)

    plt.scatter(cities['tmax'], cities['pop_density'])
    plt.title("Population Density vs Average Maximum Temperature")
    plt.xlabel('Avg Max Temperature (\u00b0C)')
    plt.ylabel('Population Density (people/km\u00b2)')

    plt.show()
    plt.savefig(output_arg)
    

main()