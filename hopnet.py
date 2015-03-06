'''
    Hopfield Neural Network Implementation in Python

    Authors: David Cunningham and Jared Smith

'''

# Built-in Python libraries
import os
import random
import contextlib
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
    requiredArguments.add_argument('-vsize', metavar='Vector_Size', dest='vsize', required=True, type=str, help='Size of vectors.')
    requiredArguments.add_argument('-nvec', metavar='Num_Vectors', dest='nvec', required=True, type=str, help='Number of vectors.')
    requiredArguments.add_argument('-nnsize', metavar='Netw_Size', dest='nnsize', required=True, type=str, help='Size of Neural Network.')
    requiredArguments.add_argument('-numruns', metavar='Num_Runs', dest= 'nruns', required = True, type=str, help='Number of runs of the experiment')

    return parser

# sigma: because who knows how many times we may have to use it
# it's that polarizing function that we use litterally all the time
def sigma(h):
        # sigma = 1 if h >= 0 and -1 if h < 0
        sigma = 0
        if h < 0:
            sigma = -1
        if h > 0:
            sigma = 1
        return sigma

# Plot the results of the combined runs using matplotlib
# I want to try to reuse this from the last lab
def plot_data(experiment_number, nvec, avg_stable_prob, avg_unstable_prob, run_no=0):

    run_str = ''
    p = list(xrange(nvec))
    abs_path = os.path.abspath(".")
    root_path = 'results/data/Experiment-' + str(experiment_number)
    file_path = 'file://' + abs_path
    if run_no == 0:
        run_str = '-run_no-' + run_no
    path = 'Graph-for-Experiment-' + experiment_number + run_str +  '.jpg'

    fig = plt.figure()

    # Plot Unstable Imprints
    plt.subplot(2, 1, 1)
    plt.plot(p, avg_unstable_prob)
    plt.legend(loc=0)
    plt.xlabel('p')
    plt.ylabel('Fraction of Unstable Imprints')
    if run_no == 0:
        plt.title('Overall Fraction of Unstable Imprints for %s Patterns' % nvec)
    else:
        plt.title('Fraction of Unstable Imprints for %s Patterns' % nvec)
    plt.grid()

    # Plot Stable Imprints
    plt.subplot(2, 1, 2)
    plt.plot(p, avg_stable_prob)
    plt.legend(loc=0)
    plt.xlabel('p')
    plt.ylabel('Fraction of Stable Imprints')
    if run_no == 0:
        plt.title('Overall Fraction of Stable Imprints for %s Patters' % nvec)
    else:
        plt.title('Fraction of Stable Imprints for %s Patters' % nvec)
    plt.grid()

    # Save the figure
    fig.savefig(root_path + '/' + path)

    return file_path + '/' + root_path + '/' + path

# Create an html page out of all the JPGs and the Graph
def create_html_page(experiment_number, graph_list, experiment_graph_path):

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
    </body>
    </html>
    ''' % (experiment_graph_path)

    html_string += string2

    with open(html_filename, 'w') as hf:
        hf.write(html_string)

class HNN:

    # Initialized at insatiation of class
    def __init__(self, args):
        self.args = args
        self.vectors = self.generate_vectors()
        self.vector_size = args.vsize
        self.num_vectors = args.nvec
        self.stable = np.zeroes((self.num_vectors)) #array of number of times a
        self.num_weights = self.vsize * self.vsize
        self.weights = np.zeroes((self.num_weights)) # would be better if we made a 100x100 matrix to copy to NN
        self.nnsize = args.nnsize
        self.NN = np.zeroes((self.nnsize))
        self.prob_stability = np.zeroes(self.num_vectors)
        self.prob_instability = np.zeroes(self.num_vectors)
        self.basin_sizes = np.zeroes(self.num_vectors, self.num_vectors)


    # Generate the patterms (vectors)
    def generate_vectors(self):
        self.vectors = []

        for m in range(self.num_vectors):
            vec = np.empty((self.vector_size))
            for n in vec:
                n = random.choice([-1, 1])
            self.vectors.append(vec)

        return self.vectors

    def calcStabilityProb(self, p):
        self.prob_stability[p] = self.stable[p]/p
        self.prob_instability[p] = 1 - self.prob_stability[p]

    def getStableProb(self):
        return self.prob_stability

    def getInstabilityProb(self):
        return self.prob_instability

    def drive(self): #driver for calculating stability (COSC 420)
        #a. generate vectors
        self.generate_vectors()
        for p in range(self.num_vectors):
            #b. imprint the first p vectors on a hopfield newtwork
            self.imprint_vectors(p)
            #c. test first p imprinted patterns for stability
            self.test_vectors(p)
            #d. Calculate stability and instability prob for each p
            self.calcStabilityProb(p)
    def getBasinHistogram(self):
        return self.basin_sizes

    # Step 1 of VanHornwender's Help
    # Check me on this, it may be completely wrong.
    def imprint_vectors(self, p):
        for i in range(nnsize):
            for j in range(nnsize):
                if i == j:
                    self.weights[self.vector_size * i + j] = 0
                else:
                    state_sum = 0
                    for k in range(p):
                        state_sum += self.vectors[k][i] * self.vectors[k][j];
                    self.weights[self.vector_size * i + j] = state_sum / self.nnsize

    #Started Step 2 of VanHornwender's Help, started to get confused here really late
    # and decided to go to bed.
    def test_vectors(self, p):
        stable_index = 0
        h_i = None
        for x in range(p):
            #1. Copy NN into pattern
            self.NN = self.vectors[x];
            new_neuron_state = 0
            stable_bool = True # keep track of stability

            #2. Compute new stat value
            for i in range(self.nnsize):
                # h_i = sum[j-1, N]{ w[i][j] * s[j] }

                for j in range(self.nnsize):
                    h_i +=  (self.weight[i * self.vector_size * i + j] * self.NN[j])
                #s'i = sigma(h)
                new_neuron_state = sigma(h_i)
                #if they don't match it wasn't stable
                if self.NN[i] != new_neuron_state:
                    stable_bool = False
                    #427/524 ONLY
                    self.basin_sizes[x][0] += 1


                self.NN[i] = new_neuron_state

            #Determine if p is stable: if so increment
            if stable_bool:
                self.stable[x] += 1
                #427/524 ONLY
                self.calc_basin_size(x)

    def calc_basin_size(self, k):
        basin = 0
        for run in range(5):
            array = np.random.permutation(self.nnsize)
            for i in range(self.num_vectors):
                self.NN = self.vectors[k]
                #flib bits for NN
                for j in range(i):
                    self.NN[array[j]] *= -1
                stable_bool = True                
                for z in range(10):    
                    for x in range(nnsize):
                        # h_i = sum[j-1, N]{ w[i][j] * s[j] }
                        for y in range(nnsize):
                            h_i +=  (self.weight[x * self.vector_size + y] * self.NN[y])
                        #s'i = sigma(h)
                        self.NN[x] = sigma(h_i)
                        #if they don't match it wasn't stable

                #if it doesn't converge after 10 runs then say it's false
                if not np.array_equal(self.NN, self.vectors[k])):
                    stable_bool = False
                    basin += i
            if stable_bool:
                basin = 50
        #average basin size            
        basin/=5

        self.basin_sizes[k][basin] += 1

# Main
if __name__ == '__main__':

    graph_list = []

    parser = setup_argparser()
    args = parser.parse_args()
    experiment_number = args.experiment_number

    if args.nnzise != args.vsize:
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
    avg_stable_prob = np.zeroes(args.nvec)
    avg_unstable_prob = np.zeroes(args.nvec)
    #do several runs of experiment compute average stability
    for i in range(args.nruns):
        hnn = HNN(args)
        hnn.stability_drive()
        stable_prob = hnn.getStableProb()
        unstable_prob = hnn.getInstabilityProb()

        graph_list += plot_data(experiment_number, args.nvec, stable_prob, unstable_prob)
        #sum stable and unstable probs
        avg_stable_prob += stable_prob
        avg_unstable_prob += unstable_prob
    #avg stable and unstable probs
    avg_stable_prob /= args.nruns
    avg_unstable_prob /= args.nruns

    #graph stable and unstable probs
    avg_graph_file = plot_data(experiment_number, args.nvec, avg_stable_prob, avg_unstable_prob)
    create_html_page(experiment_number, graph_list, avg_graph_file)

