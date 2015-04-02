# Python standard libraries
import sys
import os
import contextlib
import logging
import tempfile
import argparse
from argparse import RawTextHelpFormatter

# 3rd-Party libraries
import numpy as np
import sklearn as sk
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Project specific libraries
from network import BackPropagationNetwork
from tests import Tests
from analysis import *


# Function for changing directories safely
@contextlib.contextmanager
def cd(newPath):
    savedPath = os.getcwd()
    os.chdir(newPath)
    yield
    os.chdir(savedPath)


def is_valid_ttype_option(ttype):
    options = ['x', 'd', 'i', 'f']
    if ttype in options:
        return ttype
    else:
        print "Option 'ttype' is invalid. Choose from: "
        print options
        sys.exit(1)


# Setup the command line parser
def setup_argparser():

    parser = argparse.ArgumentParser(description='' +
                                    '    PyNet: Feed-Forward Back-Propagation Network in Python\n' +
                                    '        Written by: Jared Smith and David Cunningham',
                                     version='1.0.0', formatter_class=RawTextHelpFormatter)

    requiredArguments = parser.add_argument_group('required Arguments')
    requiredArguments.add_argument('-exp', dest='experiment_number', required=True, type=str, help="Number of this experiment.")
    requiredArguments.add_argument('-ttype', dest='test_type', required=True, type=is_valid_ttype_option, help="Type of test to run. Choose from 'x', 'd', 'i', or 'f'")
    requiredArguments.add_argument('-hidden_layers', dest='hidden_layers', required=True, type=int, nargs='+', help="A list of numbers which represent each hidden layer and the affiliate nodes in that layer.")
    optionalArguments = parser.add_argument_group('optional Arguments')
    optionalArguments.add_argument('--epochs', dest='epochs', required=False, type=int, default=2500, help="Number of epochs to train on. Default is 2500.")
    optionalArguments.add_argument('--learning_rate', dest='learning_rate', required=False, type=float, default=0.5, help="Learning rate, specifies the step width of the gradient descent. Default is 0.5.")
    optionalArguments.add_argument('--momentum_rate', dest='momentum_rate', required=False, type=float, default=0.1, help="Momentum rate, specifies the amount of the old weight change (relative to 1) which is added to the current change. Default is 0.1.")
    optionalArguments.add_argument('--learning_reward', dest='learning_reward', required=False, type=float, default=1.05, help="Magnitude to scale the learning rate by if cost/error decreases from the previous epoch. Default is 1.05.")
    optionalArguments.add_argument('--learning_penalty', dest='learning_penalty', required=False, type=float, default=0.5, help="Magnitude to scale the learning rate by if the cost/error increases from the previous epoch. Default is 0.5.")
    optionalArguments.add_argument('--regularization_term', dest='reg_term', required=False, type=float, default=1e-5, help="Regularization term (i.e. lamdba in the equations). Default is 1e-5.")
    optionalArguments.add_argument('--ftrain', dest='ftrain', required=False, type=str, default=None, help="Training data file for function approximation test. Default is None.")
    optionalArguments.add_argument('--ftest', dest='ftest', required=False, type=str, default=None, help="Testing data file for function approximation test. Default is None.")
    optionalArguments.add_argument('--fvalid', dest='fvalid', required=False, type=str, default=None, help="Validation data file for function approximation test. Default is None.")
    optionalArguments.add_argument('--plot', dest='plot', required=False, type=bool, default=True, help="Specify if data is to be plotted. Default is True.")

    return parser


def setup_logger(log_path, logger_name, logfile_name):

    logFormatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rootLogger = logging.getLogger(logger_name)
    rootLogger.setLevel(logging.DEBUG)

    fileHandler = logging.FileHandler("{0}/{1}.log".format(log_path, logfile_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    return rootLogger


if __name__=="__main__":

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

    logger = setup_logger('results/data/Experiment-' + str(experiment_number), "__main__", "main")
    logger.info("==========Experiment Number %s==========", experiment_number)
    logger.info("Program Parameters: " + str(args))

    test_suite = Tests(logger, args)
    target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse = test_suite.run_tests()



    if args.test_type != 'f':
        logger.info('Accuracy: ' + str(accuracy_score(target_test, np.rint(Y_pred).astype(int))))
        logger.info('\n' + str(classification_report(target_test, np.rint(Y_pred).astype(int))))
    else:
        target_test_1d = target_test.ravel()
        Y_pred_1d = Y_pred.ravel()
        distance = 0

        for i in range(len(target_test_1d)):
            distance += abs(Y_pred_1d[i] - target_test_1d[i])

        avg_distance = distance / len(target_test_1d)
        logger.info("Average Distance between total actual output and predicted output: %s" % (str(avg_distance)))

    save_path = 'results/data/Experiment-%s' % (experiment_number)
    save_data(save_path, target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse, experiment_number)

    # Plotting 
    if args.plot:
        plot_file_path = 'results/data/Experiment-%s' % (experiment_number)
        plot_cost_versus_epochs(plot_file_path, experiment_number, cost_list, cost_test_list)
        plot_rmse_versus_epochs(plot_file_path, experiment_number, rmse)
