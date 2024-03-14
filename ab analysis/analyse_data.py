# Utsav Anantbhat

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as py_stats
from implementations import all_implementations
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def main():
    n = len(all_implementations)
    print("Length of all implementations: " + str(n))
    combinations = (n**2 - n)/2
    combinations = int(combinations)
    print("T-test combos for n: " + str(combinations))
    print("Pr[false conclusion] = " + str(0.05/combinations))
    print("\n")

    data = pd.read_csv('data.csv')

    # Quicksort 1-5 mean
    qs1_mu = data.iloc[:,0]
    qs1_mu = qs1_mu.mean()
    qs2_mu = data.iloc[:,1]
    qs2_mu = qs2_mu.mean()
    qs3_mu = data.iloc[:,2]
    qs3_mu = qs3_mu.mean()
    qs4_mu = data.iloc[:,3]
    qs4_mu = qs4_mu.mean()
    qs5_mu = data.iloc[:,4]
    qs5_mu = qs5_mu.mean()

    # Mergesort and partition sort mean
    ms_mu = data.iloc[:,5]
    ms_mu = ms_mu.mean()
    pt_mu = data.iloc[:,6]
    pt_mu = pt_mu.mean()

    # Display mean times (seconds)
    print("qs1 mean time: " + str(qs1_mu))
    print("qs2 mean time: " + str(qs2_mu))
    print("qs3 mean time: " + str(qs3_mu))
    print("qs4 mean time: " + str(qs4_mu))
    print("qs5 mean time: " + str(qs5_mu))
    print("Merge Sort mean time: " + str(ms_mu))
    print("Partition Sort mean time: " + str(pt_mu))
    print("\n")



    # T-tests
    qs1_pt = py_stats.ttest_ind(data.iloc[:,0], data.iloc[:,6], alternative = 'greater')[1]
    qs3_ms = py_stats.ttest_ind(data.iloc[:,2], data.iloc[:,5], alternative = 'less')[1]
    qs2_ms = py_stats.ttest_ind(data.iloc[:,1], data.iloc[:,5], alternative = 'less')[1]
    qs3_4 = py_stats.ttest_ind(data.iloc[:,3], data.iloc[:,4], alternative = 'greater')[1]

    print("Quicksort 1 + Partition Sort t-test: " + str(qs1_pt))
    print("Quicksort 3 + Merge Sort t-test: " + str(qs3_ms))
    print("Quicksort 2 + Merge Sort t-test: " + str(qs2_ms))
    print("Quicksort 3 + 4 t-test: " + str(qs3_4))

    plt.xlabel('Time, seconds')
    plt.ylabel('Counts')
    #plt.legend()
    plt.title("Data Analysis (Sorting Implementations)")
    plt.hist(data.iloc[:,], bins='auto')
    plt.show()


if __name__ == '__main__':
    main()