'''
    network.py
    PyNet, Backpropagation Neural Network class

    Written by: Jared Smith and David Cunningham
'''

# Python standard libraries
import sys
import os
import tempfile
import logging
import argparse

# 3rd-Party Libraries
import sklearn as sk
import numpy as np
from scipy.special import expit

# Project files
from utils import *


# Neural Network class
class NeuralNetwork(object):

    # Class constructor
    def __init__(self, parameters):
        """ Expect parameters to be a tuple of the form:
            ((n_input,0,0), (n_hidden_layer_1, activation_function_1, deriv_activation_function_1'), ...,
            (n_hidden_layer_k, activation_function_k, deriv_activation_function_k'), (n_output, activation_function_o, deriv_activation_function_o'))
        """

        # Number of layers total including the input and output layers
        self.n_layers = len(parameters)

        # Number of nodes in each layer
        self.sizes = [layer[0] for layer in parameters]

        self.activation_functions = [layer[1] for layer in parameters]
        self.deriv_activation_functions = [layer[2] for layer in parameters]

        self.weights = []
        self.biases = []
        self.inputs = []
        self.outputs = []
        self.errors = []

        # Build the network
        self.build_network()


    # Builds the network
    def build_network(self):

        for layer in range(self.n_layers - 1):
            n = self.sizes[layer]
            m = self.sizes[layer + 1]
            self.weights.append(np.random.normal(0, 1, (m, n)))
            self.biases.append(np.random.normal(0, 1, (m, 1)))
            self.inputs.append(np.zeros((n, 1)))
            self.outputs.append(np.zeros((n, 1)))
            self.errors.append(np.zeros((n, 1)))

        n = self.sizes[-1]
        self.inputs.append(np.zeros((n, 1)))
        self.outputs.append(np.zeros((n, 1)))
        self.errors.append(np.zeros((n, 1)))


    # Propagate through the network and assign the nodes values
    def feedforward(self, x):

        k = len(x)
        x.shape = (k, 1)
        self.inputs[0] = x
        self.outputs[0] = x

        for i in range(1, self.n_layers):
            # Pay attention to the .dot function here, this does the major calculation
            self.inputs[i] = self.weights[i - 1].dot(self.outputs[i - 1]) + self.biases[i - 1]
            self.outputs[i] = self.activation_functions[i](self.inputs[i])

        y = self.outputs[-1]

        return y


    # Update the weights, errors, and bias nodes in the network
    def update_weights(self, x, y):

        output = self.feedforward(x)
        self.errors[-1] = self.deriv_activation_functions[-1](self.outputs[-1])*(output-y)
        n = self.n_layers - 2

        for i in xrange(n, 0, -1):
            self.errors[i] = self.deriv_activation_functions[i](self.inputs[i]) * self.weights[i].T.dot(self.errors[i + 1])
            self.weights[i] = self.weights[i] - self.learning_rate * np.outer(self.errors[i + 1], self.outputs[i])
            self.biases[i] = self.biases[i] - self.learning_rate * self.errors[i + 1]
        self.weights[0] = self.weights[0] - self.learning_rate * np.outer(self.errors[1], self.outputs[0])


    # Train the network, using a specific number of iterations 
    def train(self, X, y, n_iter, learning_rate=1):

        self.X = X
        self.y = y

        self.learning_rate = learning_rate
        n = self.X.shape[0]

        for repeat in range(n_iter):
            index = list(range(n))
            np.random.shuffle(index)

            for row in index:
                x = self.X[row]
                y = self.y[row]
                self.update_weights(x, y)


    # Run the neural network on the specified single input
    def predict_x(self, x):

        return self.feedforward(x)


    # Run the neural network on the specified range of input
    def predict(self, X):

        n = len(X)
        m = self.sizes[-1]
        ret = np.ones((n, m))

        for i in range(len(X)):
            ret[i, :] = self.feedforward(X[i])

        return ret

