'''
    Hopfield Neural Network Implementation in Python

    Authors: David Cunningham and Jared Smith

'''

# Built-in Python libraries
import sys
import os
import random
import datetime
import math
import contextlib
import argparse
import tempfile
from argparse import RawTextHelpFormatter

# Third party libraries
import numpy as np
from PIL import Image
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
'''def plot_data(experiment_number):

    abs_path = os.path.abspath(".")
    root_path = 'results/data/Experiment-' + str(experiment_number)
    file_path = 'file://' + abs_path
    path = 'Graph-for-Experiment-' + experiment_number + '.jpg'

    fig = plt.figure()
    axes = fig.add_subplot(111)

    axes.plot(average_p, label='Correlation (rho_l)')
    axes.plot(average_joint_h, label='Joint Entropy (H_l)')
    axes.plot(average_mutual_info, label='Mutual Info (I_l)')
    axes.legend(loc=0)
    axes.set_position((.1, .3, .8, .6))
    axes.set_xlabel('l')
    axes.set_ylabel('Values')
    axes.set_title('Experiment ' + experiment_number + ' Average Values')

    fig.text(.01, .01, description)
    plt.grid()
    fig.savefig(root_path + '/' + path)

    return file_path + '/' + root_path + '/' + path
'''

class HNN:

    # Initialized at insatiation of class
    def __init__(self, args):
        self.args = args
        self.vectors = self.generate_vectors()
        self.vector_size = args.vsize
        self.num_vectors = args.nvec
        self.stable = np.zeroes((num_vectors)) #array of number of times a
        self.num_weights = self.vsize * self.vsize
        self.weights = np.zeroes((self.num_weights)) # would be better if we made a 100x100 matrix to copy to NN
        self.nnsize = args.nnsize
        self.NN = np.zeroes((self.nnsize))
        self.prob_stability = np.zeroes(num_vectors)
        self.prob_instability = np.zeroes(num_vectors)
        self.basin_sizes = np.zeroes(num_vectors)

    # Generate the patterms (vectors)
    def generate_vectors():
        self.vectors = []

        for m in range(self.num_vectors):
            vec = np.empty((self.vector_size))
            for n in vec:
                n = random.choice([-1, 1])
            self.vectors.append(vec)

        return self.vectors

    def calcStabilityProb(p):
        self.prob_stability[p] = self.stable[p]/p
        self.prob_instability[p] = 1 - prob_stability[p]

    def getStableProb():
        return self.prob_stability

    def getInstabilityProb():
        return self.prob_instability

    def drive(): #driver for calculating stability (COSC 420)
        #a. generate vectors
        generate_vectors()
        for p in range(self.num_vectors):
            #b. imprint the first p vectors on a hopfield newtwork
            imprint_vectors(p)
            #c. test first p imprinted patterns for stability
            test_vectors(p)
            #d. Calculate stability and instability prob for each p
            calcStabilityProb(p)

    # Step 1 of VanHornwender's Help
    # Check me on this, it may be completely wrong.
    def imprint_vectors(p):
        for x in range(p):
            for i in p:
                for j in p:
                    if i == j:
                        self.weights[self.vector_size * i + j] = 0
                    else:
                        state_sum = 0
                        for i in p:
                            for j in p:
                                if i == j:
                                    self.weights[self.vector_size * i + j] = 0
                                else:
                                    state_sum += i * j;
                        self.weights[self.vector_size * i + j] = state_sum / self.vsize

    #Started Step 2 of VanHornwender's Help, started to get confused here really late
    # and decided to go to bed.
    def test_vectors(p):
        stable_index = 0
        for x in range(p):
            #1. Copy NN into pattern
            self.NN = vectors[x];
            new_neuron_state = 0
            stable_bool = True # keep track of stability

            #2. Compute new stat value
            for i in range(nnsize):
                # h_i = sum[j-1, N]{ w[i][j] * s[j] }
                for j in range(nnsize):
                    h_i +=  (weight[i * self.vector_size * i + j] * self.NN[j])
                #s'i = sigma(h)
                new_neuron_state = sigma(h_i)
                #if they don't match it wasn't stable
                if self.NN[i] != new_neuron_state:
                    stable_bool = False
                    #427/524 ONLY
                    basin_sizes[p] = 0

                self.NN[i] = new_neuron_state

            #Determine if p is stable: if so increment
            if stable_bool:
                stable[x] += 1
                #427/524 ONLY
                calc_basin_size(p)

    def calc_basin_size(p):
        for run in range(5):
            array = np.random.permutation(self.nnsize)
            for i in range(self.num_vectors):
                self.NN = vectors[p]
                doesnt_converge = False
                #flib bits for NN
                for j in range(i):
                    self.NN[array[j]] *= -1
                stable_bool = True
                converge_point = 0
                for z in range(10):    
                    for x in range(nnsize):
                        # h_i = sum[j-1, N]{ w[i][j] * s[j] }
                        for y in range(nnsize):
                            h_i +=  (weight[i * self.vector_size * i + j] * self.NN[j])
                        #s'i = sigma(h)
                        new_neuron_state = sigma(h_i)
                        #if they don't match it wasn't stable
                        if NN[i] != new_neuron_state:
                            stable_bool = False
                            basin_sizes[p] += i
                            break
                        self.NN[x] = new_neuron_state

                    if not stable_bool:
                        break

                if not stable_bool:
                    break

            if stable_bool:
                basin_size += 50
        #average basin size            
        basin_sizes[p]/=5

        #PLACE GRAPHS HERE

# Main
if __name__ == '__main__':

    image_list = []

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
        #sum stable and unstable probs
        avg_stable_prob += hnn.getStableProb()
        avg_unstable_prob += hnn.getInstabilityProb()
    #avg stable and unstable probs
    avg_stable_prob /= args.nruns
    avg_unstable_prob /= args.nruns

    #graph stable and unstable probs
