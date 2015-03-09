import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_data (data, scale): #Normalization function
    A = max(data) #max of old scale
    B = min(data) #min of old scale
    a = 0
    b = scale
    norm_data = np.copy(data)
    for x in norm_data:
        x = a + (A - x) * (b - a) / (B - A)
    return norm_data

def plot_histogram(avg_basin_size):

    (num_rows, num_cols) = avg_basin_size.shape

    fig = plt.figure()
    print "Basin of Attraction: Plotting basin probability dirstribution"
    # Histogram normalized to 1
    plt.subplot(2, 1, 1)
    for i in range(num_rows):
        label = 'p = %s' % str(i + 1)
        plt.plot(np.arange(num_cols), normalize_data(avg_basin_size[i][:], 1), label=label)
    plt.legend(loc=0)
    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probaility Distribution of Basin Sizes Normalized to 1')
    plt.grid()

    print "Basin of Attraction: Plotting basin histogram"
    # Histogram normalized to p
    plt.subplot(2, 1, 2)
    for i in range(num_rows):

        label = 'p = %s' % str(i + 1)
        plt.plot(np.arange(num_cols), normalize_data(avg_basin_size[i][:], i), label=label)

    plt.legend(loc=0)
    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probaility Distribution of Basin Sizes Normalized to P')
    plt.grid()
    fig.tight_layout()

    # Save the figure
    fig.savefig('histo_file.jpg')
    fig.show()

if __name__ == '__main__':

    histo_file = sys.argv[1]

    # try to load the 2D array fromt he file
    # since it is the only structure in the file accessing all of the
    # possible file structures should give us just that array
    histo_data = np.load(histo_file)

    plot_histogram(histo_data)



