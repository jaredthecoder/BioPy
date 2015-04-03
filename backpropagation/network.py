#!./usr/bin/python
# -*- coding: utf-8 -*-#

import numpy as np
np.set_printoptions(precision=4, suppress=True)
import math as math
from matplotlib.pyplot import plot
from sklearn.datasets import load_iris, load_digits

from utils import *

class BackPropagationNetwork(object):
    """

    Initialize as:
    nn = BackPropagationNetwork(n_features, n_classes, hidden_layers, reg_term)

    --> reg_term (i.e. lambda) is the regularization term
    nn input and output units determined by training data

    Set nn.hidden_layers to list of integers to create hidden layer architecture
    nn.hidden_layers does not include the bias unit for each layer
    nn.hidden_layers is list containing number of units in each hidden layer
        [4, 4, 2] will create 3 hidden layers of 4 units, 4 units, and 2 units
        Entire architecture will be 5 layers with units [n_features, 4, 4, 2, n_classes]

    nn.fit(X, Y, epochs) where X is training data np.array of features, Y is training data of np.array of output classes , epochs is integer specifying the number of training iterations
    For multi-class prediction, each observation in Y should be implemented as a vector with length = number of classes where each position represents a class with 1 for the True class and 0 for all other classes
    For multi-class prediction, Y will have shape n_observations by n_classes

    nn.predict(X) returns vector of probability of class being true or false
    For multi-class prediction, returns a vector for each observation will return a vector where each position in the vector is the probability of a class

    Test a simple XOR problem with nn.XOR_test()
    nn.XOR_test() accepts an optional list of integers to determine the hidden layer architecture
    """

    def __init__(self, logger, n_features, n_classes, hidden_layers, reg_term, test_type=None):
        self.logger = logger
        self.test_type = test_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.hidden_layers = hidden_layers
        self.reg_term = reg_term
        self.epochs = 2
        self.learning_rate = 0.5
        self.learning_reward = 1.05
        self.learning_penalty = 0.5
        self.momentum_rate = 0.1
        self.Theta_L = []

        self.initialize_theta()

    def initialize_theta(self):
        """
            initialize_theta creates architecture of neural network
            Defines self.Theta_L

            Parameters:
                hidden_unit_length_list - List of hidden layer units
                input_unit_count - integer, number of input units (features)
                output_class_count - integer, number of output classes
        """

        unit_count_list = [len(self.n_features[0])]
        unit_count_list += self.hidden_layers
        unit_count_list.append(len(self.n_classes[0]))
        self.Theta_L = [ 2 * (np.random.rand(unit_count, unit_count_list[l-1]+1) - 0.5) for l, unit_count in enumerate(unit_count_list) if l > 0]

    def print_theta(self):

        T = len(self.Theta_L)

        self.logger.info('\n')
        self.logger.info('NN ARCHITECTURE')
        self.logger.info('%s Layers (%s Hidden)' % ((T + 1), (T-1)))
        self.logger.info('%s Thetas' % T)
        self.logger.info('%s Input Features' % (self.Theta_L[0].shape[1]-1))
        self.logger.info('%s Output Classes' % self.Theta_L[T-1].shape[0])
        self.logger.info('\n')

        self.logger.info('Units per layer')
        for t, theta in enumerate(self.Theta_L):
            if t == 0:
                self.logger.info(' - Input: %s Units' % (theta.shape[1] - 1))
            if t < T-1:
                self.logger.info(' - Hidden %s: %s Units' % ((t+1), theta.shape[0]))
            else:
                self.logger.info(' - Output: %s Units' % theta.shape[0])

        self.logger.info('Theta Shapes')
        for l, theta in enumerate(self.Theta_L):
            self.logger.info('Theta %s: %s' % (l, theta.shape))

        self.logger.info('Theta Values')
        for l, theta in enumerate(self.Theta_L):
            self.logger.info('Theta %s:' % l)
            self.logger.info("\n" + str(theta))
        self.logger.info('\n')

    def cost_function(self, Y, Y_pred):
        """
        cost_function implements cost function

        y is n_observations by n_classes (n_classes = 1 for n_classes <=2)
        pred_y is predicted y values and must be same shape as y

        Returns cost - list of cost values
        """

        if Y.shape != Y_pred.shape:
            if Y.shape[0] != Y_pred.shape:
                raise ValueError,'Wrong number of predictions'
            else:
                raise ValueError,'Wrong number of prediction classes'

        n_observations = len(Y)
        tiny = 1e-6
        # Cost Function
        cost = (-1.0 / n_observations)*(Y * np.log(Y_pred + tiny) + ((1-Y) * np.log(1-Y_pred + tiny))).sum()

        return cost


    def predict(self, X):
        """
        predict calculates activations for all layers, returns prediction for Y

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
            Theta_Gradient_L[t] = Theta_Gradient_L[t] * (1.0/n_observations) + (self.reg_term * regTheta[t])

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
            learning_reward
            learning_penalty
            X_test - np.array of training data with dimension n_observations by n_features
            Y_test - np.array of training classes with dimension n_observations by n_classes
        Returns
            cost_list - list of result of cost function for each epoch
            Learning_rates - list of learning rates used for each epoch
        Notes
            Training and Test data are assumed to be in random order; mini-batch processing does not need to re-randomize
        """

        # Initial Learning Rate
        learning_rates = []
        learning_rates.append( self.learning_rate )

        # Initial Weight Change Terms
        weight_change_L = [np.zeros_like(theta) for theta in self.Theta_L]

        # List of results of cost functions
        cost_list = [0] * self.epochs
        cost_test_list = [0] * self.epochs
        rmse = [0] * self.epochs
        # Initial Forward Pass
        a_N_train, Y_pred = self.predict(X_train)
        # Initial Cost
        cost_list[0] = self.cost_function(Y_train, Y_pred)

        # Test Error
        if Y_test is not None:
            a_N_test, Y_pred_test = self.predict(X_test)
            cost_test_list[0] = self.cost_function(Y_test, Y_pred_test)

        for i in range(1, self.epochs):

            # Back Prop to get Theta Gradients
            Theta_grad = self.back_prop(a_N_train, Y_train)

            # Update Theta with Momentum
            for l, theta_g in enumerate(Theta_grad):
                weight_change_L[l] = self.learning_rate * theta_g + (weight_change_L[l] * self.momentum_rate)
                self.Theta_L[l] = self.Theta_L[l] - weight_change_L[l]

            # Update Units
            a_N_train, Y_pred_new = self.predict(X_train)

            # Check to see if Cost decreased
            cost_new = self.cost_function(Y_train, Y_pred_new)

            if cost_new > cost_list[i-1]:
                # Reduce learning rate
                self.learning_rate = self.learning_rate * self.learning_penalty
                # Reverse part of adjustment (add back new learning_rate * Theta_grad); Leave momentum in place
                self.Theta_L = [t + (self.learning_rate * tg) for t, tg in zip(self.Theta_L, Theta_grad)]
                # Cut prior weight_change as an approximate fix to momentum
                weight_change_L = [m * self.learning_penalty for m in weight_change_L]

                a_N_train, Y_pred_new = self.predict(X_train)
                cost_new = self.cost_function(Y_train, Y_pred_new)
            else:
                self.learning_rate = np.min((10, self.learning_rate * self.learning_reward))

            learning_rates.append(self.learning_rate)
            cost_list[i] = cost_new

            if Y_test is not None:
                a_N_test, Y_pred_test = self.predict(X_test)
                
                sum_e = 0
                for j in range(len(Y_test)):
                    sum_e += pow((Y_test[j] - Y_pred_test[j]), 2)

                if len(sum_e) > 1:
                    sum_e = np.sum(sum_e)

                rmse_epoch = math.sqrt((1.0 / (2.0 * len(Y_test))) * sum_e)
                rmse[i] = rmse_epoch
                
                cost_test_list[i] = self.cost_function(Y_test, Y_pred_test)

        for t, theta in enumerate(self.Theta_L):
            self.logger.info('Theta: %s' % t)
            for theta_i in np.round(theta, 2):
                self.logger.info("%s" % str(theta_i))   

        #self.logger.info('i: %ld - cost:      %ld' %  (i, cost_list[i]))
        #self.logger.info('i: %ld - cost test: %ld' %  (i, cost_test_list[i]))

        return cost_list, learning_rates, cost_test_list, rmse

    def test(self, data_train, target_train, epochs, learning_rate, momentum_rate, learning_reward, learning_penalty, data_test=None, target_test=None, data_val=None, target_val=None):

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.learning_reward = learning_reward
        self.learning_penalty = learning_penalty

        # Initialize Theta based on selected architecture
        # self.initialize_theta(data_train.shape[1], target_train.shape[1], hidden_unit_length_list)

        # Fit
        cost_list, learning_rates, cost_test_list, rmse = self.fit(data_train, target_train, X_test=data_test, Y_test=target_test)

        error = 0
        # Predict for test log
        plot_vals = []
        a_N, Y_pred = self.predict(data_test)
        self.logger.info('###################################Testing Results###################################')
        self.logger.info('Given X:')
        for x in data_test[:5]:
            self.logger.info(x)
        for p in zip(target_test[:10], np.round(Y_pred[:10],6)):
            plot_vals.append(p)
        self.logger.info('Actual Y, Predicted Y:')
        for pv in plot_vals:
            self.logger.info("%s" % str(pv))
        self.logger.info('Cost Efficiency on Test Set: %s' % str(self.cost_function(target_test , Y_pred)))
        sum_e = 0
        for j in range(len(target_test)):
            sum_e += pow((target_test[j] - Y_pred[j]), 2)
        if len(sum_e) > 1:
            sum_e = np.sum(sum_e)
        self.logger.info('Final Testing Sum Over Outputs: %s' % str(sum_e))
        rmse_test_final = math.sqrt((1.0 / (2.0 * len(target_test))) * sum_e)
        self.logger.info('Final Testing RMSE: %s' % str(rmse_test_final))

        error = 0
        #Predict for validation results

        if data_val is not None:
            plot_vals = []
            va_N, vY_pred = self.predict(data_val)
            self.logger.info('###################################Validation Results###############################')
            self.logger.info('Given X:')
            for x in data_val[:5]:
                self.logger.info(x)
            for p in zip(target_val[:10], np.round(vY_pred[:10],6)):
                plot_vals.append(p)
            self.logger.info('Actual Y, Predicted Y:')
            for pv in plot_vals:
                self.logger.info((pv))
            self.logger.info('Cost Efficiency on Validation Set: %s' % str(self.cost_function(target_val , vY_pred)))
            sum_e = 0
            for j in range(len(target_val)):
                sum_e += pow((target_val[j] - vY_pred[j]), 2)
            if len(sum_e) > 1:
                sum_e = np.sum(sum_e)
            self.logger.info('Final Validation Sum Over Outputs: %s' % str(sum_e))
            rmse_val_final = math.sqrt((1.0 / (2.0 * len(target_val))) * sum_e)
            self.logger.info('Final Validation RMSE: %s' % str(rmse_val_final))

        return target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse

