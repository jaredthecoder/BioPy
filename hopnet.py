'''
    hopnet.py
    Hopfield Neural Network Implmentation in Python

    Written by: Jared Smith and David Cunningham

    How to get up and running:

        1) Make a virtualenv with 'virtualenv venv' and 'source venv/bin/activate'
           If you don't already have the command 'virtualenv', then you can install
           virtualenv with pip.
        2) Install program dependencies with 'pip install -r requirements.txt'
        3) Run 'python hopnet.py -h' to see the required arguments for running the program.

'''


# Built-in Python libraries
import os
import random
import contextlib
import sys
import argparse
from argparse import RawTextHelpFormatter

# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

# Initialize the random seed
random.seed()

# Function for changing directories safely
@contextlib.contextmanager
def cd(newPath):
    savedPath = os.getcwd()
    os.chdir(newPath)
    yield
    os.chdir(savedPath)

# Setup the command line parser
def setup_argparser():

    parser = argparse.ArgumentParser(description='' +
                                    'Written by: Jared Smith and David Cunningham.',
                                     version='1.0.0', formatter_class=RawTextHelpFormatter)
    requiredArguments = parser.add_argument_group('required Arguments')
    requiredArguments.add_argument('-exp', metavar='Exp_Num', dest='experiment_number', required=True, type=str, help='Number of this experiment.')
    requiredArguments.add_argument('-npat', metavar='Num_Patterns', dest='npat', required=True, type=int, help='Number of patterns.')
    requiredArguments.add_argument('-nnsize', metavar='Netw_Size', dest='nnsize', required=True, type=int, help='Size of Neural Network.')
    requiredArguments.add_argument('-nruns', metavar='Num_Runs', dest= 'nruns', required =True, type=int, help='Number of runs of the experiment.')
    requiredArguments.add_argument('-dfn', metavar='Data_File_Name', dest= 'dfn', required =True, type=str, help='Data file name to save experiment data to.')
    requiredArguments.add_argument('-hfn', metavar='Histo_File_Name', dest= 'hfn', required =True, type=str, help='Data file name to save histogram data to.')


    return parser

# Get Peak of Data Curve
def getpeak(data):
    peak_y = np.max(data)
    peak_x = np.argmax(data)
    return peak_x, peak_y

# Normalize Data
def normalize_data (data, scale, p):
    norm_data = data.copy()
    b = np.min(norm_data)
    norm_data -= b
    norm_data /= p
    norm_data *= scale

    return norm_data

# Make the graphs for each run and the overall average of runs
def plot_graph_data(experiment_number, nvec, avg_stable_prob, avg_unstable_prob, run_no):

    run_str = ''
    p = list(xrange(nvec))

    abs_path = os.path.abspath(".")
    root_path = 'results/data/Experiment-' + str(experiment_number)
    file_path = 'file://' + abs_path
    if run_no == 0:
        run_str = ''
    else:
        run_str = '-runnum-%s-' % str(run_no)
    path = 'Graph-for-Experiment-' + experiment_number + run_str +  '.jpg'

    fig = plt.figure()

    # Plot Unstable Imprints
    plt.subplot(2, 1, 1)
    plt.plot(p, avg_unstable_prob)
    plt.xlabel('p')
    plt.ylabel('Fraction of Unstable Imprints')
    if run_no == 0:
        plt.title('Overall Fraction of Unstable Imprints for %s Patterns' % nvec)
    else:
        plt.title('Fraction of Unstable Imprints for %s Patterns' % nvec)
    plt.legend(loc=0)
    plt.grid()

    # Plot Stable Imprints
    plt.subplot(2, 1, 2)
    plt.plot(p, avg_stable_prob)
    plt.xlabel('p')
    plt.ylabel('Fraction of Stable Imprints')
    if run_no == 0:
        plt.title('Overall Fraction of Stable Imprints for %s Patters' % nvec)
    else:
        plt.title('Fraction of Stable Imprints for %s Patters' % nvec)
    plt.legend(loc=0)
    plt.grid()
    fig.tight_layout()

    # Save the figure
    fig.savefig(root_path + '/' + path)

    return file_path + '/' + root_path + '/' + path

# Plot the Histogram
# CS427/527 ONLY
def plot_histogram(avg_basin_size, experiment_number):

    (num_rows, num_cols) = avg_basin_size.shape
    avg_basin_size[:][:] += 1

    abs_path = os.path.abspath(".")
    root_path = 'results/data/Experiment-' + str(experiment_number)
    file_path = 'file://' + abs_path
    path = 'Histogram-for-Experiment-' + experiment_number + '.jpg'

    fig = plt.figure()

    # Plot the Histogram Normalized to 1
    plt.subplot(2, 1, 1)
    for i in range(1, num_rows + 1):
        if i % 2 == 1:
            text_str = '%d' % ((i + 1))
            n = normalize_data(avg_basin_size[i-1][:], 1, i)
            peak_x, peak_y = getpeak(n)
            plt.plot(np.arange(0, num_cols), n)

            # Label the curve
            if peak_y < 1.0 and peak_x > 1:
                plt.text(peak_x, peak_y+.1, text_str)

    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probability Distribution of Basin Sizes Normalized to 1')
    plt.grid()
    plt.ylim(0, 1.3)

    # Plot the Histogram Normalized to 1
    plt.subplot(2, 1, 2)
    for i in range(1, num_rows + 1):
        if i % 2 == 1:
            text_str = '%d' % ((i + 1))
            n = normalize_data(avg_basin_size[i-1][:], i, i)
            peak_x, peak_y = getpeak(n)
            plt.plot(np.arange(0, num_cols), n)

            # Label the Curve
            if peak_y < 4.3 and peak_x > 1:
                plt.text(peak_x, peak_y+.1, text_str)

    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probability Distribution of Basin Sizes Normalized to P')
    plt.grid()
    plt.ylim(0,4.5)

    # Fix layout issues in plot.
    fig.tight_layout()

    # Save the figure
    fig.savefig(root_path + '/' + path)

    return file_path + '/' + root_path + '/' + path

# Handle Invalid weights
class InvalidWeightsException(Exception):
    pass

# Handle Invalid NN Input
class InvalidNetworkInputException(Exception):
    pass

# Class for maintaining all of the data structures used for the simulation
class Data(object):

    # Initialize Class
    def __init__(self, args, histo_file=None):
        self._nnsize = args.nnsize
        self._npat = args.npat
        self._exp = args.experiment_number
        self._stable = np.zeros(self._npat)
        self._basin_hist = np.zeros((self._npat, self._npat + 1))
        self._prunstable = np.copy(self._stable)
        self._data_file_name = args.dfn
        self._histo_data_file_name = histo_file

    # Calculate the probablity vectors
    def calc_prob(self):
        stable_prob = np.zeros(self._npat)
        for p in range(self._npat):
            stable_prob[p] = self._stable[p] / (p+1)
            self._prunstable[p] = 1 - stable_prob[p]

    # Sum each run
    def sum(self, data):
        self._stable += data._stable
        self._prunstable += data._prunstable
        self._basin_hist += data._basin_hist

    # Average all of the runs
    def avg(self, nruns):
        self._stable /= nruns
        self._basin_hist /= nruns
        self._prunstable /= nruns

    # Save the report data as a human readable text file
    # Can also be read back into NumPy Arrays or Python Lists
    def save_report_data(self):
        with file(self._data_file_name, 'w') as outfile:
            outfile.write('# Average Stable Probability Data\n')
            outfile.write('# Array shape {0}\n'.format(self._stable))
            np.savetxt(outfile, self._stable, fmt='%-7.2f')
            outfile.write('\n')

            outfile.write('# Average Unstable Probability Data\n')
            outfile.write('# Array shape {0}\n'.format(self._prunstable))
            np.savetxt(outfile, self._prunstable, fmt='%-7.2f')
            outfile.write('\n')

            outfile.write('# Average Basin Histogram Data\n')
            outfile.write('# Array shape {0}\n'.format(self._basin_hist))
            np.savetxt(outfile, self._basin_hist, fmt='%-7.2f')
            outfile.write('\n')

    # Save Histogram data for use in plotting separately
    def save_histo_data(self):
        np.save(self._histo_data_file_name, self._basin_hist)

    # Graph each run
    def graph(self, run):
        return plot_graph_data(self._exp, self._npat, self._stable, self._prunstable, run)

# Class for the Hopfield Neural Network Model used
class HopfieldNetwork(object):

    # Initialize Class
    def __init__(self, num_inputs):
        self._num_inputs = num_inputs
        self._weights = np.random.uniform(-1.0, 1.0, (num_inputs,num_inputs))

    # Set the weights of the weight vector
    def set_weights(self, weights):
        if weights.shape != (self._num_inputs, self._num_inputs):
            raise InvalidWeightsException()

        self._weights = weights

    # Get the weights
    def get_weights(self):
        return self._weights

    # Evaluate the state of the Neural Network
    def evaluate(self, input_pattern):
        if input_pattern.shape != (self._num_inputs, ):
            raise InvalidNetworkInputException()

        weights = np.copy(self._weights)

        # Compute the sums of the input patterns
        # Uses dot product
        sums = input_pattern.dot(weights)

        s = np.zeros(self._num_inputs)

        # Enumerate the sums of the inputs and calculate new values
        for i, value in enumerate(sums):
            if value > 0:
                s[i] = 1
            else:
                s[i] = -1

        return s

    # Run the updating of the Network for a specified iteration count
    # Default number of iterations is 10
    def run(self, input_pattern, iterations=10):
        last_input_pattern = input_pattern
        iteration_count = 0

        while True:
            result = self.evaluate(last_input_pattern)

            iteration_count += 1
            if  np.array_equal(result, last_input_pattern) or iteration_count == iterations:
                return result
            else:
                last_input_pattern = result

# Imprint the patterns onto the network
def imprint_patterns(network, input_patterns, p):
    num_neurons = network.get_weights().shape[0]
    weights = np.zeros((num_neurons, num_neurons))

    for i in range(num_neurons):
        for j in range(num_neurons):
            if i == j: continue
            for m in range(p):
                weights[i, j] += input_patterns[m][i] * input_patterns[m][j]

    weights *= 1 / float(network._num_inputs)

    network.set_weights(weights)

# Test the patterns for stability
def test_patterns(p, input_patterns, network, data):
    for k in range(p):
        pattern = input_patterns[k][:]
        updated_pattern = np.copy(pattern)
        updated_pattern = network.run(updated_pattern, 1)
        if np.array_equal(updated_pattern, pattern):
            data._stable[p - 1] +=1

            # Run Basin test
            data = basin_test(p, pattern, network, data, 5)
        else:
            data._basin_hist[p - 1][0] += 1

    return data

# Run the basin size tests
def basin_test(p, input_pattern, network, data, runs):
    basin = 0

    for run in range(runs):
        converge = True
        array =  np.random.permutation(data._nnsize)
        updated_pattern = np.copy(input_pattern)

        for i in range(1, data._npat + 1):
            for j in range (i):
                updated_pattern[array[j]] *= -1

            updated_pattern = network.run(updated_pattern)
            if not np.array_equal(updated_pattern, input_pattern):
                converge = False
                basin += i
                break

        if converge:
            basin += 50

    basin = round((basin / runs), 0)
    data._basin_hist[p - 1][basin] += 1
    return data

# Run the experiment
def run_experiment(args):
    stable = np.zeros((args.npat))
    input_patterns = np.random.choice([-1,1], ((args.npat), (args.nnsize)))
    Hnet = HopfieldNetwork((args.nnsize))
    data = Data(args)

    # Go through all npat patterns and run tests
    for p in range (1, args.npat + 1):
        imprint_patterns(Hnet, input_patterns, p)
        test_patterns(p, input_patterns, Hnet, data)
    data.calc_prob()

    return data

# Execute on program start
if __name__ == '__main__':

    graph_list = []

    parser = setup_argparser()
    args = parser.parse_args()
    experiment_number = args.experiment_number
    histo_file = 'results/data/Experiment-%s/%s' % (experiment_number, args.hfn)

    # Setup directories for storing results
    if not os.path.exists('results'):
        os.makedirs('results')

    with cd('results'):
        if not os.path.exists('data'):
            os.makedirs('data')
        with cd('data'):
            if not os.path.exists('Experiment-' + str(experiment_number)):
                os.makedirs('Experiment-' + str(experiment_number))

    avg_data = Data(args, histo_file)

    # Run experiment for nruns number of times
    for i in range(1, int(args.nruns) + 1):
        exp_data = run_experiment(args)
        graph_list += exp_data.graph(i)
        avg_data.sum(exp_data)

    avg_data.avg(args.nruns)

    # Save the average data
    avg_data.save_report_data()
    avg_data.save_histo_data()

    # Plot the average stability graphs and make the histogram for basin sizes
    plot_graph_data(experiment_number, args.npat, avg_data._stable, avg_data._prunstable, 0)
    plot_histogram(avg_data._basin_hist, experiment_number)


