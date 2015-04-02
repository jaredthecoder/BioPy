import sys

import numpy as np
import math as math
from matplotlib.pyplot import plot
from sklearn.datasets import load_iris, load_digits

from network import BackPropagationNetwork
from utils import *

class Tests(object):

    def __init__(self, logger, args):

        self.logger = logger
        self.args = args

    def run_tests(self):
        if self.args.test_type == 'x':
            self.logger.info("###################################RUNNING XOR TEST##################################")
            target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse = self.XOR_test(self.args.reg_term, self.args.hidden_layers, self.args.epochs, self.args.learning_rate, self.args.momentum_rate, self.args.learning_reward, self.args.learning_penalty)
            return (target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse)
        elif self.args.test_type == 'd':
            self.logger.info("###################################RUNNING DIGITS TEST###############################")
            target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse = self.digits_test(self.args.reg_term, self.args.hidden_layers, self.args.epochs, self.args.learning_rate, self.args.momentum_rate, self.args.learning_reward, self.args.learning_penalty)
            return (target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse)
        elif self.args.test_type == 'i':
            self.logger.info("###################################RUNNING IRIS TEST#################################")
            target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse = self.iris_test(self.args.reg_term, self.args.hidden_layers, self.args.epochs, self.args.learning_rate, self.args.momentum_rate, self.args.learning_reward, self.args.learning_penalty)
            return (target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse)
        elif self.args.test_type == 'f':
            self.logger.info("###################################RUNNING DIGITS TEST###############################")
            target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse = self.fnct_aprox(self.args.reg_term, self.args.hidden_layers, self.args.epochs, self.args.learning_rate, self.args.momentum_rate, self.args.learning_reward, self.args.learning_penalty, self.args.ftrain, self.args.ftest, self.args.fvalid)
            return (target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse)


    def translate_to_binary_array(self, target):
        n_obs = len(target)
        unique_targets = np.unique(target)
        n_unique_targets = len(np.unique(target))

        # Translation of target values to array indicies
        target_translation = dict(zip(unique_targets, range(n_unique_targets)))

        # Create initial target array with all zeros
        target_array = np.zeros((n_obs, n_unique_targets))

        # Set 1 value
        for i, val in enumerate(target):
            target_array[i][target_translation[val]] = 1

        return target_array

    def train_test_split(self, data_array, target_array, split=.8):
        """
        Split into randomly shuffled train and test sets
        Split on Number of records or Percent of records in the training set
        if split is <= 1 then split is a percent, else split is the number of records
        """

        n_obs = len(data_array)

        if split <= 1:
            train_len = int(split * n_obs)
        else:
            train_len = int(np.round(split))

        shuffled_index = range(n_obs)
        np.random.shuffle(shuffled_index)

        train_data = data_array[shuffled_index[:train_len]]
        test_data = data_array[shuffled_index[train_len:]]

        train_target = target_array[shuffled_index[:train_len]]
        test_target = target_array[shuffled_index[train_len:]]

        self.logger.info('Data Set: %d Observations, %d Features' % (data_array.shape[0], data_array.shape[1]))
        self.logger.info('Training Set: %d Observations, %d Features' % (train_data.shape[0], train_data.shape[1]))
        self.logger.info('Test Set: %d Observations, %d Features' % (test_data.shape[0], test_data.shape[1]))
        self.logger.info('Target Set: %d Observations, %d Classes' % (target_array.shape[0], target_array.shape[1]))
        self.logger.info('Training Set: %d Observations, %d Features' % (train_target.shape[0], train_target.shape[1]))
        self.logger.info('Test Set: %d Observations, %d Features' % (test_target.shape[0], test_target.shape[1]))

        return train_data, test_data, train_target, test_target

    def iris_test(self, reg_term, hidden_layers, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup):

        data_set = load_iris()

        data = data_set.data
        target = self.translate_to_binary_array(data_set.target)

        # Split into train, test sets
        data_train, data_test, target_train, target_test = self.train_test_split(data, target, .75)

        NN = BackPropagationNetwork(self.logger, data_train, target_train, hidden_layers, reg_term)
        return BackPropagationNetwork.test(NN, data_train, target_train, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)

    def digits_test(self, reg_term, hidden_layers, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup):

        data_set = load_digits()

        data = data_set.data
        target = self.translate_to_binary_array(data_set.target)

        # Split into train, test sets
        data_train, data_test, target_train, target_test = self.train_test_split(data, target, .75)

        NN = BackPropagationNetwork(self.logger, data_train, target_train, hidden_layers, reg_term)
        return BackPropagationNetwork.test(NN, data_train, target_train, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)

    def XOR_test(self, reg_term, hidden_layers, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup):
        """
        XOR_test is a simple test of the nn self.logger.info(ing the predicted value to std out
        Trains on a sample XOR data set
        Predicts a single value
        Accepts an option parameter to set architecture of hidden layers
        """

        # Set Data for XOR Test
        data_train = np.zeros((4,2))
        data_train[0,0] = 1.
        data_train[1,1] = 1.
        data_train[2,:] = 1.
        data_train[3,:] = 0.

        target_train = np.array([1.,1.,0.,0.]).reshape(4,1)         # Single Class

        # Test X and Y
        data_test = np.array([[1,0],[0,1],[1,1],[0,0]])
        target_test = np.array([[1],[1],[0],[0]])

        self.logger.info('Training Data: X')
        for data_i in data_train:
            self.logger.info("%s" % str(data_i))
        self.logger.info('Training Data: Y')
        for target_i in target_train:
            self.logger.info("%s" % str(target_i))

        NN = BackPropagationNetwork(self.logger, data_train, target_train, hidden_layers, reg_term)
        return BackPropagationNetwork.test(NN, data_train, target_train, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)

    def fnct_aprox(self, reg_term, hidden_layers, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, training_name, testing_name, validation_name):

        #read in train
        data_train, target_train = self.parse_file(training_name, 200)
        np.random.shuffle(data_train)
        np.random.shuffle(target_train)
        #read in test
        data_test, target_test = self.parse_file(testing_name, 100)
        np.random.shuffle(data_test)
        np.random.shuffle(target_test)
        #read in validation
        data_val, target_val = self.parse_file(validation_name, 50)
        np.random.shuffle(data_val)
        np.random.shuffle(target_val)

        NN = BackPropagationNetwork(self.logger, data_train, target_train, hidden_layers, reg_term)
        return BackPropagationNetwork.test(NN, data_train, target_train, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test, data_val = data_val,target_val = target_val)

    def parse_file(self, filename, num_lines):
        data = []
        target = []
        f = open(filename, 'r')
        for line in f:
            floats = map(float, line.split())
            target.append([floats.pop()])
            data.append(floats)
        f.close()
        return np.array(data), np.array(target)
