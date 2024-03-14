# Utsav Anantbhat


import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import statsmodels.api as stats
import pykalman as pk


infile = sys.argv[1] #input the CSV file as a command line argument
cpu_data = pd.read_csv(infile)

# Plot data for testing

plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
#plt.show() # maybe easier for testing


# LOESS Smoothing

lowess = stats.nonparametric.lowess #define lowess
frac = 0.04 #set the frac parameter value
cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp']) #convert timestamp to a pandas datetime object
cpu_temp = cpu_data['temperature'].astype('Float64')
datetime = cpu_data['timestamp'].apply(np.datetime64) #get the date in a numpy array in the format of 'yyyy-mm-dd'; reference: https://stackoverflow.com/questions/13703720/converting-between-datetime-timestamp-and-datetime64
loess_smoothed = lowess(cpu_temp, datetime, frac = frac)

plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')
#plt.show()


# Kalman Smoothing

kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']] # keep these columns from the data

# Kalman filter parameters:
initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([0.72, 0.72, 0.72, 0.72]) ** 2 # TODO: shouldn't be zero
transition_covariance = np.diag([0.07, 0.07, 0.07, 0.07]) ** 2 # TODO: shouldn't be zero
transition = [[0.94,0.5,0.2,-0.001], [0.1,0.4,2.1,0], [0,0,0.94,0], [0,0,0,1]] # TODO: shouldn't (all) be zero; values based on predictions as stated in the assignment

# Smooth Kalman data:
kf = pk.KalmanFilter(initial_state_mean = initial_state, 
                     observation_covariance = observation_covariance, 
                     transition_covariance = transition_covariance,
                     transition_matrices = transition)

kalman_smoothed, _ = kf.smooth(kalman_data)
plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
#plt.show()


# Plot all figures:
plt.figure(figsize=(12, 4))
plt.title('CPU Temperature Noise Reduction')
plt.xlabel('Time')
plt.ylabel('Temperature(C)')
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
plt.show()
plt.savefig('cpu.svg')