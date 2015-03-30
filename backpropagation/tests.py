'''
    tests.py
    PyNet, Backpropagation Neural Network class

    Written by: Jared Smith and David Cunningham
'''


# Python standard libraries
import sys
import os
import contextlib
import tempfile
import argparse
from argparse import RawTextHelpFormatter

# 3rd-Party libraries
import sklearn as sk
import numpy as np
from scipy.special import expit

# Project specific libraries
from network import *
from utils import *


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
    requiredArguments.add_argument('-exp', dest='experiment_number', required=True, type=str, help='Number of this experiment.')
    requiredArguments.add_argument('-ttype', dest='test_type', required=True, type=str, help="Type of test to run. Choose 'r' " +
                                    "for regression or 'c' for classification.")

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


# Try to approximate a sine curve with a linear regression
def test_regression(plots=False):

    predictions = []

    n = 200
    learning_rate = 0.05

    X = np.linspace(0, 3 * np.pi, num=n)
    X.shape = (n, 1)
    y = np.sin(X)

    # Make the network structure parameters, see network.py parameters variable for how to structure
    layers = ((1, 0, 0), (20, expit, logistic_prime), (20, expit, logistic_prime), (1, identity, identity_prime))

    # Run the network
    N = NeuralNetwork(X, y, layers)
    N.train(4000, learning_rate=learning_rate)
    predictions.append([learning_rate, N.predict(X)])


    # Plotting 
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)

    if plots:
        ax.plot(X, y, label='Sine', linewidth=2, color='black')
        for data in predictions:
            ax.plot(X, data[1], label="Learning Rate: " + str(data[0]))
        ax.legend()
        plt.show()


# Try to approximate a sine curve with a non-linear classification
def test_classification(plots=False):
    
    # Number of samples
    n = 700

    n_iter = 1500
    learning_rate = 0.05

    # Samples for true decision boundary plot
    L = np.linspace(0,3 * np.pi, num=n)
    l = np.sin(L)

    # Data inputs, training
    X = np.random.uniform(0, 3 * np.pi, size=(n, 2))
    X[:, 1] *= 1 / np.pi
    X[:, 1] -= 1


    # Data inputs, testing
    T = np.random.uniform(0, 3 * np.pi, size=(n, 2))
    T[:, 1] *= 1 / np.pi
    T[:, 1] -= 1

    # Data outputs
    y = np.sin(X[:, 0]) <= X[:, 1]

    # Fitting
    param = ((2, 0, 0), (30, expit, logistic_prime), (30, expit, logistic_prime), (1, expit, logistic_prime))
    N = NeuralNetwork(X, y, param)

    # Training and Validation
    N.train(n_iter, learning_rate)
    predictions_training = N.predict(X)
    predictions_training = predictions_training < 0.5
    predictions_training = predictions_training[:, 0]

    # Testing
    predictions_testing = N.predict(T)
    predictions_testing = predictions_testing < 0.5
    predictions_testing = predictions_testing[:, 0]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)

    # Training plot

    # We plot the predictions of the neural net blue for class 0, red for 1.
    ax[0].scatter(X[predictions_training, 0], X[predictions_training, 1], color='blue')
    not_index = np.logical_not(predictions_training)
    ax[0].scatter(X[not_index, 0], X[not_index, 1], color='red')
    ax[0].set_xlim(0, 3 * np.pi)
    ax[0].set_ylim(-1, 1)

    # True decision boundary
    ax[0].plot(L, l, color='black')
    
    # Shade the areas according to how to they should be classified.
    ax[0].fill_between(L, l, y2=-1, alpha=0.5)
    ax[0].fill_between(L, l, y2=1, alpha=0.5, color='red')

    # Testing plot
    ax[1].scatter(T[predictions_testing, 0], T[predictions_testing, 1], color='blue')
    not_index = np.logical_not(predictions_testing)
    ax[1].scatter(T[not_index, 0], T[not_index, 1], color='red')
    ax[1].set_xlim(0, 3 * np.pi)
    ax[1].set_ylim(-1, 1)
    ax[1].plot(L, l, color='black')
    ax[1].fill_between(L, l, y2=-1, alpha=0.5)
    ax[1].fill_between(L, l, y2=1, alpha=0.5, color='red')

    if plots:
        plt.show()


if __name__ == "__main__":


    graph_list = []

    parser = setup_argparser()
    args = parser.parse_args()
    experiment_number = args.experiment_number

    # Setup directories for storing results
    if not os.path.exists('results'):
        os.makedirs('results')

    with cd('results'):
        if not os.path.exists('data'):
            os.makedirs('data')
        with cd('data'):
            if not os.path.exists('Experiment-' + str(experiment_number)):
                os.makedirs('Experiment-' + str(experiment_number))

    test_type = args.test_type

    if test_type == 'r':
        test_regression(True)
    elif test_type == 'c':
        test_classification(True)
    else:
        print "Bad Command. Try 'r' or 'c'"


