import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_data (data, scale): #Normalization function
    arr = np.copy(data)
    np_minmax = ((arr - arr.min()) / (arr.max() - arr.min())) * scale
    return np_minmax

def plot_histogram(avg_basin_size):

    (num_rows, num_cols) = avg_basin_size.shape
    avg_basin_size[:][:] += 1
    #bins = np.arange(0, num_cols)

    fig = plt.figure()
    # Histogram normalized to 1
    plt.subplot(2, 1, 1)
    for i in range(1, num_rows + 1):
        if i % 2 == 0:
            label = 'p = %s' % str(i + 1)
            plt.hist(avg_basin_size[i - 1][:], num_cols, normed=True, alpha=0.5)
    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probability Distribution of Basin Sizes Normalized to 1')
    #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=10)
    plt.grid()

    # Histogram normalized to p
    plt.subplot(2, 1, 2)
    for i in range(1, num_rows + 1):
        if i % 2 == 0:
            label = 'p = %s' % str(i + 1)
            plt.hist(avg_basin_size[i - 1][:], num_cols, normed=True, alpha=0.5)
    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probability Distribution of Basin Sizes Normalized to P')
    plt.grid()
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



