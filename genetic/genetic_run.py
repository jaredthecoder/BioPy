##############################################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# genetic.py, general purpose genetic algorithm in Python
#
# Written by Jared Smith and David Cunningham for COSC 427/527
# at the University of Tennessee, Knoxville
#
###############################################################

import os
import sys
import argparse
import json
import math
import logging
import tempfile
import random
from argparse import RawTextHelpFormatter

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

from genetic_algorithms import BaseGeneticAlgorithm

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
                                    '    PyNet: General Purpose Genetic Algorithm in Python\n' +
                                    '        Written by: Jared Smith and David Cunningham',
                                     version='1.0.0', formatter_class=RawTextHelpFormatter)

    requiredArguments = parser.add_argument_group('required Arguments')
    requiredArguments.add_argument('-exp', dest='experiment_number', required=True, type=str, help="Number of this experiment.")
    optionalArguments = parser.add_argument_group('optional Arguments')
    optionalArguments.add_argument('--num_bits', dest='l', required=False, type=int, default=20, help="Number of bits (genes) in the genetic string. Default is 20.")
    optionalArguments.add_argument('--population_size', dest='N', required=False, type=int, default=30, help="Size of the population. Default is 30.")
    optionalArguments.add_argument('--num_gens', dest='G', required=False, type=int, default=10, help="Number of generations. Default is 10.")
    optionalArguments.add_argument('--pr_mutation', dest='pm', required=False, type=Float, default=0.033, help="Probability of Mutation. Default is 0.033.")
    optionalArguments.add_argument('--pr_crossover', dest='pc', required=False, type=Float, default=0.6, help="Probability of Crossover. Default is 0.6.")
    optionalArguments.add_argument('--plot', dest='plot', required=False, type=bool, default=True, help="Specify if data is to be plotted. Default is True.")
    optionalArguments.add_argument('--autoscale', dest='autoscale', required=False, type=bool, default=True, help="Specify plots should be autoscaled to data frame. Default is True.")
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


def main():
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
    logger.info("###################################RUNNING EXPERIMENT NUM %s#########################", str(experiment_number))
    logger.info("Program Arguments:")
    args_dict = vars(args)
    for key, value in args_dict.iteritems() :
        logger.info("%s=%s" % (str(key), str(value)))


    logger.info("Running Base Genetic Algorithm...")
    gen_alg = BaseGeneticAlgorithm(args, logger)
    gen_alg.run()

    if args.plot:
        logger.debug("Going to plot, but don't have data left.")

if __name__=="__main__":
    main()
