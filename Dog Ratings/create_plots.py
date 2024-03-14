# Utsav Anantbhat


import matplotlib.pyplot as plt
import pandas as pd
import sys

filename1 = sys.argv[1]
filename2 = sys.argv[2]

arg_1 = pd.read_csv(filename1, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])

arg_2 = pd.read_csv(filename2, sep=' ', header=None, index_col=1,
        names=['lang', 'page', 'views', 'bytes'])


arg_1.sort_values(by=['views'], inplace=True, ascending=False)

wiki_views = arg_1['views'].values
plt.figure(figsize=(10, 5)) # change the size to something sensible
plt.subplot(1, 2, 1) # subplots in 1 row, 2 columns, select the first
plt.title("Plot 1: Distribution of Views")
plt.xlabel("Rank")
plt.ylabel("Views")
plt.plot(wiki_views) # build plot 1

arg_2['views_arg_2'] = arg_1['views']
plt.subplot(1, 2, 2) # ... and then select the second
plt.title("Plot 2: Hourly Views")
plt.xlabel("Hour 1 Views")
plt.ylabel("Hour 2 Views")
plt.xscale("log")
plt.yscale("log")
plt.scatter(arg_2['views_arg_2'], arg_2['views'], marker='.') # build plot 2

plt.savefig('wikipedia.png')