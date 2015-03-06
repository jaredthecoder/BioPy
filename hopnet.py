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
    requiredArguments.add_argument('-vsize', metavar='Vector_Size', dest='vsize', required=True, type=str, help='Size of p vectors.')
    requiredArguments.add_argument('-nvec', metavar='Num_Vectors', dest='nvec', required=True, type=str, help='Number of p vectors.')
    requiredArguments.add_argument('-nnsize', metavar='Netw_Size', dest='nnsize', required=True, type=str, help='Size of Neural Network.')



    return parser

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
        self.num_weights = self.vsize * self.vsize
        self.weights = np.zeroes((self.num_weights))
        self.nnsize = args.nnsize
        self.NN = np.zeroes((self.nnsize))

    # Generate the patterms (vectors)
    def generate_vectors():
        self.vectors = []

        for m in range(self.num_vectors):
            vec = np.empty((self.vector_size))
            for n in vec:
                n = random.choice([-1, 1])
            self.vectors.append(vec)

        return self.vectors

    # Step 1 of VanHornwender's Help
    # Check me on this, it may be completely wrong.
    def imprint_vectors():

        for p in self.vectors:
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


    # Started Step 2 of VanHornwender's Help, started to get confused here really late
    # and decided to go to bed.
    def test_vectors():

        for p in self.vectors:
            self.NN = p
            for i in p:
                for j in self.NN:


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

    hnn = HNN(args)
    hnn.imprint_vectors()
    hnn.test_vectors()



