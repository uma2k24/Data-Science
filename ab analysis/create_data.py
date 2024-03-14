# Utsav Anantbhat

import sys
import numpy as np
import pandas as pd
import time
from implementations import all_implementations

def main():

    n = 10000 #large n
    sample = n/100
    sample = int(sample)

    out = np.zeros([sample, len(all_implementations)])

    for i in np.arange(sample):
        random_array = np.random.randint(1000, 10*n, size = (n,))

        for sort in np.arange(len(all_implementations)):
            st = time.time()
            res = all_implementations[sort](random_array)
            en = time.time()
            time_taken = en-st

            out[i][sort] = time_taken
    
    data = pd.DataFrame(out, columns=[all_implementations])
    data.to_csv('data.csv', index=False)


if __name__ == '__main__':
    main()