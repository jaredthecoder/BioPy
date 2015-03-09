import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab

#this is the finalized plot histogram function
def getpeak(data):
    peak_y = np.max(data)
    peak_x = np.argmax(data)
    return peak_x, peak_y

def normalize_data (data, scale, p): #Normalization function
    norm_data = data.copy()
    b = np.min(norm_data)
    norm_data -= b #set bottom to zero
    norm_data /= p #get probability distribution
    norm_data *= scale

    return norm_data

def plot_histogram(avg_basin_size):

    (num_rows, num_cols) = avg_basin_size.shape
    avg_basin_size[:][:] += 1

    fig = plt.figure()
    # Histogram normalized to 1
    plt.subplot(2, 1, 1)
    for i in range(1, num_rows + 1):
        if i % 2 == 1:
            text_str = 'p = %s' % str(i + 1)
            n = normalize_data(avg_basin_size[i-1][:], 1, i)
            peak_x, peak_y = getpeak(n)
            plt.plot(np.arange(0, num_cols), n)
            #label
            if peak_y < 1.0 and peak_x > 1:
                plt.text(peak_x, peak_y+.1, text_str)

    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probability Distribution of Basin Sizes Normalized to 1')
    #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=10)
    plt.grid()
    plt.ylim(0, 1.3)

    # Histogram normalized to p
    plt.subplot(2, 1, 2)
    for i in range(1, num_rows + 1):
        if i % 2 == 1:
            text_str = 'p = %s' % str(i + 1)
            n = normalize_data(avg_basin_size[i-1][:], i, i)
            peak_x, peak_y = getpeak(n)
            plt.plot(np.arange(0, num_cols), n)
            #lable
            if peak_y < 4.3 and peak_x > 1:
                plt.text(peak_x, peak_y+.1, text_str)

    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probability Distribution of Basin Sizes Normalized to P')
    plt.grid()
    plt.ylim(0,4.5)
    #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=10)

    # Save the figure
    fig.tight_layout()
    fig.savefig('histo_file.jpg')

if __name__ == '__main__':

    histo_file = sys.argv[1]

    # try to load the 2D array fromt he file
    # since it is the only structure in the file accessing all of the
    # possible file structures should give us just that array
    histo_data = np.load(histo_file)
    plot_histogram(histo_data)



