#!./usr/bin/python
# -*- coding: utf-8 -*-#

import numpy as np
from matplotlib.pyplot import plot
from sklearn.datasets import load_iris, load_digits

from utils import *

class BackPropagationNetwork(object):
    """
    py_nn() implements a python neural net with back propagation used to fit the theta parameters.
    Uses Numpy as np
    Uses matplotlib.pyplot.plot

    Initialize:    nn = py_nn()

    nn input and output units determined by training data

    Set nn.hidden_layer_length_list to list of integers to create hidden layer architecture
    nn.hidden_layer_length_list does not include the bias unit for each layer
    nn.hidden_layer_length_list is list containing number of units in each hidden layer
        [4, 4, 2] will create 3 hidden layers of 4 units, 4 units, and 2 units
        Entire architecture will be 5 layers with units [n_features, 4, 4, 2, n_classes]

    nn.fit(X, Y, epochs) where X is training data np.array of features, Y is training data of np.array of output classes , epochs is integer specifying the number of training iterations
    For multi-class prediction, each observation in Y should be implemented as a vector with length = number of classes where each position represents a class with 1 for the True class and 0 for all other classes
    For multi-class prediction, Y will have shape n_observations by n_classes

    nn.nn_predict(X) returns vector of probability of class being true or false
    For multi-class prediction, returns a vector for each observation will return a vector where each position in the vector is the probability of a class

    Test a simple XOR problem with nn.XOR_test()
    nn.XOR_test() accepts an optional list of integers to determine the hidden layer architecture
    """

    def __init__(self):

        self.Theta_L = [] 			# List of Theta numpy.arrays
        self.lmda = 1e-5			# Regularization term
        self.hidden_layer_length_list = []
        self.reset_theta() 			# Sets self.hidden_layer_length_list to [2]
        self.epochs = 2
        self.learning_rate = 0.5
        self.learning_acceleration = 1.05
        self.learning_backup = 0.5
        self.momentum_rate = 0.1

    def reset_theta(self):
        """self.reset_theta sets theta as a single hidden layer with 2 hidden units"""
        self.hidden_layer_length_list = [2]

    def initialize_theta(self, input_unit_count, output_class_count, hidden_unit_length_list):
        """
            initialize_theta creates architecture of neural network
            Defines self.Theta_L

            Parameters:
                hidden_unit_length_list - List of hidden layer units
                input_unit_count - integer, number of input units (features)
                output_class_count - integer, number of output classes
        """

        if not hidden_unit_length_list:
            hidden_unit_length_list = self.hidden_layer_length_list
        else:
            self.hidden_layer_length_list = hidden_unit_length_list

        unit_count_list = [input_unit_count]
        unit_count_list.extend(hidden_unit_length_list)
        unit_count_list.append(output_class_count)
        self.Theta_L = [ 2 * (np.random.rand( unit_count, unit_count_list[l-1]+1 ) - 0.5) for l, unit_count in enumerate(unit_count_list) if l > 0]


    def print_theta(self):
        """print_theta(self) prints self.Theta_L and architecture info to std out"""

        T = len(self.Theta_L)

        print
        print 'NN ARCHITECTURE'
        print '%s Layers (%s Hidden)' % ((T + 1), (T-1))
        print '%s Thetas' % T
        print '%s Input Features' % (self.Theta_L[0].shape[1]-1)
        print '%s Output Classes' % self.Theta_L[T-1].shape[0]
        print

        print 'Units per layer'
        for t, theta in enumerate(self.Theta_L):
            if t == 0:
                print ' - Input: %s Units' % (theta.shape[1] - 1)
            if t < T-1:
                print ' - Hidden %s: %s Units' % ((t+1), theta.shape[0])
            else:
                print ' - Output: %s Units' % theta.shape[0]
        print

        print 'Theta Shapes'
        for l, theta in enumerate(self.Theta_L):
            print 'Theta %s: %s' % (l, theta.shape)
        print

        print 'Theta Values'
        for l, theta in enumerate(self.Theta_L):
            print 'Theta %s:' % l
            print theta
        print

    def nn_cost(self, Y, Y_pred):
        """
        nn_cost implements cost function

        y is n_observations by n_classes (n_classes = 1 for n_classes <=2)
        pred_y is predicted y values and must be same shape as y

        Returns J - list of cost values
        """
        Y.shape
        Y_pred.shape
        if Y.shape != Y_pred.shape:
            if Y.shape[0] != Y_pred.shape:
                raise ValueError,'Wrong number of predictions'
            else:
                raise ValueError,'Wrong number of prediction classes'

        n_observations = len(Y)
        tiny = 1e-6
        # Cost Function
        J = (-1.0/n_observations)*(Y * np.log(Y_pred + tiny) + ((1-Y) * np.log(1-Y_pred + tiny))).sum()

        return J


    def nn_predict(self, X):
        """
        nn_predict calculates activations for all layers, returns prediction for Y

        Parameters
            X is array of input features dimensions n_observations by n_features

        Returns
            a_N is outputs of all units
            a_N[L] is array of predicted Y values dimensions n_observations by n_classes
        """

        m = len(X)
        T = len(self.Theta_L)

        a_N_predict = []		# List of activations including bias unit for non-output layers

        # Input Layer inputs
        a_N_predict.append( X )
        # Loop through each Theta_List theta
        # t is Theta for calculating layer t+1 from layer t
        for t, theta in enumerate(self.Theta_L):
            # Add bias unit
            if a_N_predict[t].ndim == 1:
                a_N_predict[t].resize(1, a_N_predict[t].shape[0])
            a_N_predict[t] = np.append(np.ones((a_N_predict[t].shape[0],1)), a_N_predict[t], 1)

            # Calculate and Append new z and a arrays to z_N and a_N lists
            z = a_N_predict[t].dot(theta.T)
            a_N_predict.append(sigmoid(z))

        return a_N_predict, a_N_predict[T]


    def back_prop(self, a_N_backprop, Y_train):
        """
        a_N - list of layer outputs with dimensions n_observations by n_units
        Y_train is n_observations, n_classes

        Returns
            Theta_Gradient_L
        """
        T = len(self.Theta_L)
        Y_pred = a_N_backprop[T]
        n_observations = len(Y_pred)

        # Backprop Error; One list element for each layer
        delta_N = []

        # Get Error for Output Layer
        delta = Y_pred - Y_train
        if delta.ndim == 1:
            delta.resize(1, len(delta))
        delta_N.append( delta )

        # Get Error for Hidden Layers working backwards (stop before layer 0; no error in input layer)
        for t in range(T-1,0,-1):
            delta = delta.dot(self.Theta_L[t][:,1:]) * ( a_N_backprop[t][:,1:] * (1 - a_N_backprop[t][:,1:]) )
            delta_N.append( delta )
        # Reverse the list so that delta_N[t] is delta that Theta[t] causes on a_N[t+1]
        delta_N.reverse()

        # Calculate Gradient from delta and activation
        # t is the Theta from layer t to layer t+1
        Theta_Gradient_L = []
        for t in range(T):
            Theta_Gradient_L.append( delta_N[t].T.dot(a_N_backprop[t]) )

        # Create modified copy of the Theta_L for Regularization
        # Coefficient for theta values from bias unit set to 0 so that bias unit is not regularized
        regTheta = [np.zeros_like(theta) for theta in self.Theta_L]
        for t, theta in enumerate(self.Theta_L):
            regTheta[t][:,1:] = theta[:,1:]

        # Average Error + regularization penalty
        for t in range(T):
            Theta_Gradient_L[t] = Theta_Gradient_L[t] * (1.0/n_observations) + (self.lmda * regTheta[t])

        return Theta_Gradient_L

    def fit(self, X_train, Y_train, X_test=None, Y_test=None):
        """
        fit() calls the predict and back_prop functions for the
        given number of cycles, tracks error and error improvement rates

        Parameters:
            X_train - np.array of training data with dimension n_observations by n_features
            Y_train - np.array of training classes with dimension n_observations by n_classes
            epochs -    integer of number of times to update Theta_L
            learning_rate
            momentum_rate
            learning_acceleration
            learning_backup
            X_test - np.array of training data with dimension n_observations by n_features
            Y_test - np.array of training classes with dimension n_observations by n_classes
        Returns
            J_list - list of result of cost function for each epoch
            Learning_rates - list of learning rates used for each epoch
        Notes
            Training and Test data are assumed to be in random order; mini-batch processing does not need to re-randomize
        """

        # If no Theta provided, use a 3 layer architecture with hidden_layer units = 2 or y classes or x features
        if not self.Theta_L:
            hidden_units = max(2, len(Y_train[0]), len(X_train[0]))
            self.initialize_theta(len(X_train[0]), len(Y_train[0]), [hidden_units])

        # Initial Learning Rate
        learning_rates = []
        learning_rates.append( self.learning_rate )

        # Initial Weight Change Terms
        weight_change_L = [np.zeros_like(theta) for theta in self.Theta_L]

        # List of results of cost functions
        J_list = [0] * self.epochs
        J_test_list = [0] * self.epochs

        # Initial Forward Pass
        a_N_train, Y_pred = self.nn_predict(X_train)
        # Initial Cost
        J_list[0] = self.nn_cost(Y_train, Y_pred)

        # Test Error
        if Y_test is not None:
            a_N_test, Y_pred_test = self.nn_predict(X_test)
            J_test_list[0] = self.nn_cost(Y_test, Y_pred_test)

        for i in range(1,self.epochs):

            # Back Prop to get Theta Gradients
            Theta_grad = self.back_prop(a_N_train, Y_train)

            # Update Theta with Momentum
            for l, theta_g in enumerate(Theta_grad):
                weight_change_L[l] = self.learning_rate * theta_g + (weight_change_L[l] * self.momentum_rate)
                self.Theta_L[l] = self.Theta_L[l] - weight_change_L[l]

            # Update Units
            a_N_train, Y_pred_new = self.nn_predict(X_train)

            # Check to see if Cost decreased
            J_new = self.nn_cost(Y_train, Y_pred_new)

            if J_new > J_list[i-1]:
                # Reduce learning rate
                self.learning_rate = self.learning_rate * self.learning_backup
                # Reverse part of adjustment (add back new learning_rate * Theta_grad); Leave momentum in place
                self.Theta_L = [t + (self.learning_rate * tg) for t, tg in zip(self.Theta_L, Theta_grad)]
                # Cut prior weight_change as an approximate fix to momentum
                weight_change_L = [m * self.learning_backup for m in weight_change_L]

                a_N_train, Y_pred_new = self.nn_predict(X_train)
                J_new = self.nn_cost(Y_train, Y_pred_new)
            else:
                self.learning_rate = np.min((10,self.learning_rate * self.learning_acceleration))

            learning_rates.append(self.learning_rate)
            J_list[i] = J_new

            if Y_test is not None:
                a_N_test, Y_pred_test = self.nn_predict(X_test)
                J_test_list[i] = self.nn_cost(Y_test, Y_pred_test)

        for t, theta in enumerate(self.Theta_L):
            print 'Theta: %s' % t
            print np.round(theta, 2)

        print 'i:',i,'    - J:',J_list[i]
        print 'i:',i,'    - J test:',J_test_list[i]

        return J_list, learning_rates, J_test_list

    def nn_test(self, data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test=None, target_test=None):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.learning_acceleration = learning_acceleration
        self.learning_backup = learning_backup

        # Initialize Theta based on selected architecture
        self.initialize_theta(data_train.shape[1], target_train.shape[1], hidden_unit_length_list)

        # Fit
        J_list, learning_rates, J_test_list = self.fit(data_train, target_train, X_test=data_test, Y_test=target_test)

        # Predict
        a_N, Y_pred = self.nn_predict(data_test)
        print 'Given X:'
        print data_test[:5]
        print 'Actual Y, Predicted Y:'
        for p in zip(target_test[:10], np.round(Y_pred[:10],3)):
            print p
        print
        print 'CE on Test Set'
        print self.nn_cost(target_test , Y_pred)

        return target_test, Y_pred, J_list, J_test_list, learning_rates