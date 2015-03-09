'''
    Hopfield Neural Network Implementation in Python

    Authors: David Cunningham and Jared Smith

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
    requiredArguments.add_argument('-vsize', metavar='Vector_Size', dest='vsize', required=True, type=int, help='Size of vectors.')
    requiredArguments.add_argument('-nvec', metavar='Num_Vectors', dest='nvec', required=True, type=int, help='Number of vectors.')
    requiredArguments.add_argument('-nnsize', metavar='Netw_Size', dest='nnsize', required=True, type=int, help='Size of Neural Network.')
    requiredArguments.add_argument('-numruns', metavar='Num_Runs', dest= 'nruns', required = True, type=int, help='Number of runs of the experiment')

    return parser

def normalize_data (data, scale): #Normalization function
    A = max(data) #max of old scale
    B = min(data) #min of old scale
    a = 0
    b = scale
    norm_data = data.copy()
    for x in norm_data:
        x = a + (A - x) * (b - a) / (B - A)
    return norm_data

# Plot the results of the combined runs using matplotlib
# I want to try to reuse this from the last lab
def plot_graph_data(experiment_number, nvec, avg_stable_prob, avg_unstable_prob, run_no):

    run_str = ''
    p = list(xrange(int(nvec)))
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
    print "Ploting Unstable Imprints"
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
    print "Plotting Stable Imprints"
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

def plot_histogram(experiment_number, avg_basin_size):

    (num_rows, num_cols) = avg_basin_size.shape

    abs_path = os.path.abspath(".")
    root_path = 'results/data/Experiment-' + str(experiment_number)
    file_path = 'file://' + abs_path
    path = 'Histogram-for-Experiment-' + experiment_number + '.jpg'

    fig = plt.figure()
    print "Basin of Attraction: Plotting basin probability dirstribution"
    # Histogram normalized to 1
    plt.subplot(2, 1, 1)
    for i in range(num_rows):
        label = 'p = %s' % str(i + 1)
        plt.plot(np.arange(num_cols), normalize_data(avg_basin_size[:][i], 1), label=label)
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
        plt.plot(np.arange(num_cols), normalize_data(avg_basin_size[:][i], i), label=label)

    plt.legend(loc=0)
    plt.xlabel('B')
    plt.ylabel('Value')
    plt.title('Probaility Distribution of Basin Sizes Normalized to P')
    plt.grid()
    fig.tight_layout()

    # Save the figure
    fig.savefig(root_path + '/' + path)

    return file_path + '/' + root_path + '/' + path

# Create an html page out of all the JPGs and the Graph
def create_html_page(experiment_number, graph_list, histofile, avg_graph_file):
    print "Creating html page"

    html_filename = 'results/data/Experiment-' + str(experiment_number) + '/Experiment-' + experiment_number + '.html'

    html_string = ''

    string1 = '''
    <html>
    <head>
    <title>Experiment %s</title>
    </head>
    <body>
    <h1>Experiment %s</h1>
    <ul>
    ''' % (experiment_number, experiment_number)

    html_string += string1
    for graph_path in graph_list:
        graph_path_list = graph_path.split("/")
        graph_header = graph_path_list[-1]
        tmp_string = "<h3>%s<h3>" % (graph_header)
        tmp_string += "<li><img src='%s' height=300 width=300 border=1></li>" % (graph_path)
        html_string += tmp_string

    string2 ='''
    </ul>
    <h2>Average of Runs -- Data Graph</h2>
    <img src='%s'height=300 width=300 border=1>
    <h2>Basin of Attraction Histograms</h2>
    <img src='%s'height=600 width=300 border=1>
    </body>
    </html>
    ''' % (avg_graph_file, histofile)

    html_string += string2

    with open(html_filename, 'w') as hf:
        hf.write(html_string)

class HNN:

    # Initialized at insatiation of class
    def __init__(self, args):
        print "Initializing Hopfiled"
        self.args = args
        self.vector_size = args.vsize
        self.num_vectors = args.nvec
        self.vectors = np.random.choice([-1,1], (self.num_vectors, self.vector_size))
        self.stable = np.zeros((self.num_vectors)) #array of number of times a
        self.weights = np.zeros((self.vector_size, self.vector_size)) # would be better if we made a 100x100 matrix to copy to NN
        self.nnsize = args.nnsize
        self.NN = np.zeros((self.nnsize))
        self.prob_stability = np.zeros(self.num_vectors)
        self.prob_instability = np.zeros(self.num_vectors)
        self.basin_sizes = np.zeros((self.num_vectors, self.num_vectors))

    def calcStabilityProb(self, p):
        print "Calculating Probability of stability"
        self.prob_stability[p] = self.stable[p] / (p+1)
        self.prob_instability[p] = 1 - self.prob_stability[p]

    def getStableProb(self):
        return self.prob_stability

    def getInstabilityProb(self):
        return self.prob_instability

    def drive(self): #driver for calculating stability (COSC 420)
        for p in range(self.num_vectors):
            print 'p = {}'.format(p+1)
            #a. imprint the first p vectors on a hopfield newtwork
            self.imprint_vectors(p)
            #b. test first p imprinted patterns for stability
            self.test_vectors(p)
            #c. Calculate stability and instability prob for each p
            self.calcStabilityProb(p)

    def getBasinSizes(self):
        return self.basin_sizes

    # Step 1 of VanHornwender's Help
    # Check me on this, it may be completely wrong.
    def imprint_vectors(self, p):
        print "Imprinting vectors"
        for i in range(self.nnsize):
            for j in range(self.nnsize):
                if i == j:
                    self.weights[i][j] = 0
                else:
                    state_sum = 0
                    for k in range(p+1):
                        state_sum += (self.vectors[k][i] * self.vectors[k][j]);
                    self.weights[i][j] = state_sum / self.nnsize

    # sigma: because who knows how many times we may have to use it
    # it's that polarizing function that we use litterally all the time
    def sigma(self, h):
        # sigma = 1 if h >= 0 and -1 if h < 0
        sigma = 0
        if h <= 0:
            sigma = -1
        if h > 0:
            sigma = 1
        return sigma
    #Started Step 2 of VanHornwender's Help, started to get confused here really late
    # and decided to go to bed.
    def test_vectors(self, p):
        print "Testing vectors for stability"
        for k in range(p+1):
            #1. Copy NN into pattern
            self.NN = np.copy(self.vectors[k][:])
            new_neuron_state = 0
            stable_bool = True # keep track of stability

            #2. Compute new stat value
            for i in range(self.nnsize):
                # h_i = sum[j-1, N]{ w[i][j] * s[j] }
                h_i = 0
                for j in range(self.nnsize):
                    h_i += (self.weights[i][j] * self.NN[j])
                #s'i = sigma(h)
                new_neuron_state = self.sigma(h_i)
                #if they don't match it wasn't stable
                if self.NN[i] != new_neuron_state:
                    stable_bool = False
                    #427/524 ONLY
                    self.basin_sizes[p][0] += 1


                self.NN[i] = new_neuron_state

            #Determine if p is stable: if so increment
            if stable_bool:
                self.stable[k] += 1
                #427/524 ONLY
                self.calc_basin_size(k, p)

    def calc_basin_size(self, k, p):
        print "Calculating basin of attraction"
        basin = 0
        h_i = 0
        stable_bool = None
        cur_pattern = np.copy(self.vectors[k][:])
        for run in range(5):
            array = np.random.permutation(self.nnsize)
            for i in range(1,self.num_vectors+1):
                self.NN= np.copy(self.vectors[k][:])

                # flip bits for NN
                print 'flipping {} bits'.format(i)
                for j in range(i):
                    self.NN[array[j]] *= -1
                stable_bool = True
                print "Testing if it still converges in 10 runs"
                for z in range(10):
                    for x in range(self.nnsize):
                        # h_i = sum[j-1, N]{ w[i][j] * s[j] }
                        for y in range(self.nnsize):
                            h_i +=  (self.weights[x][y] * self.NN[y])
                        # s'i = sigma(h)
                        self.NN[x] = self.sigma(h_i)
                        # if they don't match it wasn't stable

                # if it doesn't converge after 10 runs then say it's false
                if not np.array_equal(self.NN, cur_pattern):
                    print 'It does not converges after fliping {} bits'.format(i)
                    stable_bool = False
                    basin += i
                    break

            if stable_bool:
                print 'It still converge after flipping 50 bits'
                basin = 50

        # average basin size
        basin /= 5

        self.basin_sizes[p][round(basin, 0)] += 1


# Main
if __name__ == '__main__':

    graph_list = []

    np.set_printoptions(threshold=np.nan)

    parser = setup_argparser()
    args = parser.parse_args()
    experiment_number = args.experiment_number

    if args.nnsize != args.vsize:
        print "Size of Neural Network and Size of Pattern Vector must be the same."
        exit(1)

    # Setup directories for storing results
    if not os.path.exists('results'):
        os.makedirs('results')


    with cd('results'):
        if not os.path.exists('data'):
            os.makedirs('data')
        with cd('data'):
            if not os.path.exists('Experiment-' + str(experiment_number)):
                os.makedirs('Experiment-' + str(experiment_number))

    # compute program and graph
    # initialize avg stability
    print(args.nvec + 3)

    avg_stable_prob = np.zeros((args.nvec))
    avg_unstable_prob = np.zeros((args.nvec))
    avg_basin_size = np.zeros((args.nvec, args.nvec))

    #do several runs of experiment compute average stability
    for i in range(1, args.nruns + 1):
        print 'Run {}'.format(i)
        hnn = HNN(args)
        hnn.drive()
        stable_prob = hnn.getStableProb()
        unstable_prob = hnn.getInstabilityProb()
        basin_sizes = hnn.getBasinSizes()

        graph_list += plot_graph_data(experiment_number, args.nvec, stable_prob, unstable_prob, i)

        #sum stable and unstable probs
        avg_stable_prob += stable_prob
        avg_unstable_prob += unstable_prob
        avg_basin_size += basin_sizes

    #avg stable and unstable probs
    print "Averaging ..."
    avg_stable_prob /= args.nruns
    avg_unstable_prob /= args.nruns
    avg_basin_size /= args.nruns

    #graph stable and unstable probs
    avg_graph_file = plot_graph_data(experiment_number, int(args.nvec), avg_stable_prob, avg_unstable_prob, 0)
    histo_file = plot_histogram(experiment_number, avg_basin_size)

    create_html_page(experiment_number, graph_list, histo_file, avg_graph_file)

