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
import argparse
import logging
import contextlib
from argparse import RawTextHelpFormatter

import numpy as np
import matplotlib.pyplot as plt

from genetic_algorithms import BaseGeneticAlgorithm
from ffs import fitness_func_1


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
                                     '    PyNet: General Purpose Genetic ' +
                                     ' Algorithm in Python\n' +
                                     '        Written by: Jared Smith and ' +
                                     ' David Cunningham',
                                     version='1.0.0',
                                     formatter_class=RawTextHelpFormatter)

    requiredArguments = parser.add_argument_group('required Arguments')
    requiredArguments.add_argument('-exp', dest='experiment_number', required=True, type=str, help="Number of this experiment.")
    optionalArguments = parser.add_argument_group('optional Arguments')
    optionalArguments.add_argument('--num_bits', dest='l', required=False, type=int, default=20, help="Number of bits (genes) in the genetic string. Default is 20.")
    optionalArguments.add_argument('--population_size', dest='N', required=False, type=int, default=30, help="Size of the population. Default is 30.")
    optionalArguments.add_argument('--num_gens', dest='G', required=False, type=int, default=10, help="Number of generations. Default is 10.")
    optionalArguments.add_argument('--pr_mutation', dest='pm', required=False, type=float, default=0.033, help="Probability of Mutation. Default is 0.033.")
    optionalArguments.add_argument('--pr_crossover', dest='pc', required=False, type=float, default=0.6, help="Probability of Crossover. Default is 0.6.")
    optionalArguments.add_argument('--learn_offspring', dest='learn', required=False, type=bool, default=False, help="Specify whether to enforce learning on the offspring of each generation. Default is False.")
    optionalArguments.add_argument('--change_environment', dest='ce', required=False, type=bool, default=False, help="Specify whether to inflict a sudden change of environment on the final population. Default is False.")
    optionalArguments.add_argument('--num_learning_guesses', dest='NG', required=False, type=int, default=20, help="Specify the number of guesses to take when learning with the offspring. Default is 20.")
    optionalArguments.add_argument('--fitness_func', dest='ff', required=False, type=callable, default=fitness_func_1, help="Specify the fitness function to use. Default is fitness_func_1 from utils.py.")
    optionalArguments.add_argument('--plot', dest='plot', required=False, type=bool, default=True, help="Specify if data is to be plotted. Default is True.")
    optionalArguments.add_argument('--autoscale', dest='autoscale', required=False, type=bool, default=True, help="Specify plots should be autoscaled to data frame. Default is True.")
    optionalArguments.add_argument('--nruns', dest='nruns', required=False, type=int, default=10, help="Specify the number of runs to do of the algorithm. Default is 10.")

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


def plot_results(G, nruns, avg_fitness_vals, best_fitness_vals, num_correct_bits,
                 autoscale, plot_file_path, experiment_number, ce):
    x = np.arange(1, G)

    # Plot average fitness values
    fig = plt.figure()
    subplt = plt.subplot(111)

    for nrun in nruns:
        plt.plot(x, avg_fitness_vals[nrun][0])
    plt.xlabel('Generations')
    plt.ylabel('Average Fitness Value')
    plt.title('Average Fitness Values Over %d Runs with %d Generations' %
              (avg_fitness_vals.shape[0], avg_fitness_vals.shape[2]))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True, True, True)
    fig.tight_layout()
    plot_file_name = "%s/avg-fit-vals-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    # Plot the best fitness values
    fig = plt.figure()
    subplt = plt.subplot(111)

    for nrun in nruns:
        plt.plot(x, best_fitness_vals[nrun][0])
    plt.xlabel('Generations')
    plt.ylabel('Best Fitness Value')
    plt.title('Best Fitness Values Over %d Runs with %d Generations' %
              (best_fitness_vals.shape[0], best_fitness_vals.shape[2]))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True, True, True)
    fig.tight_layout()
    plot_file_name = "%s/best-fit-vals-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    # Plot the number of correct bits for the best individual
    fig = plt.figure()
    subplt = plt.subplot(111)

    for nrun in nruns:
        plt.plot(x, num_correct_bits[nrun][0])
    plt.xlabel('Generations')
    plt.ylabel('Number of Correct Bits')
    plt.title('Number of Correct Bits Over %d Runs with %d Generations' %
              (num_correct_bits.shape[0], num_correct_bits.shape[2]))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True, True, True)
    fig.tight_layout()
    plot_file_name = "%s/num-correct-bits-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    if ce:
        # Plot average fitness values
        fig = plt.figure()
        subplt = plt.subplot(111)

        for nrun in nruns:
            plt.plot(x, avg_fitness_vals[nrun][1])
        plt.xlabel('Generations')
        plt.ylabel('Average Fitness Value')
        plt.title('CE Average Fitness Values Over %d Runs with %d Generations' %
                (avg_fitness_vals.shape[0], avg_fitness_vals.shape[2]))
        plt.grid()
        if autoscale:
            subplt.autoscale_view(True, True, True)
        fig.tight_layout()
        plot_file_name = "%s/ce-avg-fit-vals-exp-%s.pdf" % (plot_file_path, experiment_number)
        plt.savefig(plot_file_name, bbox_inches='tight')

        # Plot the best fitness values
        fig = plt.figure()
        subplt = plt.subplot(111)

        for nrun in nruns:
            plt.plot(x, best_fitness_vals[nrun][1])
        plt.xlabel('Generations')
        plt.ylabel('Best Fitness Value')
        plt.title('CE Best Fitness Values Over %d Runs with %d Generations' %
                (best_fitness_vals.shape[0], best_fitness_vals.shape[2]))
        plt.grid()
        if autoscale:
            subplt.autoscale_view(True, True, True)
        fig.tight_layout()
        plot_file_name = "%s/ce-best-fit-vals-exp-%s.pdf" % (plot_file_path, experiment_number)
        plt.savefig(plot_file_name, bbox_inches='tight')

        # Plot the number of correct bits for the best individual
        fig = plt.figure()
        subplt = plt.subplot(111)

        for nrun in nruns:
            plt.plot(x, num_correct_bits[nrun][1])
        plt.xlabel('Generations')
        plt.ylabel('Number of Correct Bits')
        plt.title('CE Number of Correct Bits Over %d Runs with %d Generations' %
                (num_correct_bits.shape[0], num_correct_bits.shape[2]))
        plt.grid()
        if autoscale:
            subplt.autoscale_view(True, True, True)
        fig.tight_layout()
        plot_file_name = "%s/ce-num-correct-bits-exp-%s.pdf" % (plot_file_path, experiment_number)
        plt.savefig(plot_file_name, bbox_inches='tight')


def main():
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

    logger = setup_logger('results/data/Experiment-'
                          + str(experiment_number), "__main__", "main")
    logger.info("###################################RUNNING EXPERIMENT NUM " +
                "%s#########################", str(experiment_number))
    logger.info("Program Arguments:")
    args_dict = vars(args)
    for key, value in args_dict.iteritems():
        logger.info("%s=%s" % (str(key), str(value)))

    logger.info("Running Base Genetic Algorithm...")
    gen_alg = BaseGeneticAlgorithm(args, logger)
    avg_fitness_vals, best_fitness_vals, num_correct_bits = gen_alg.run()
    logger.info("Finished Base Genetic Algorithm.")

    if args.plot:
        plot_file_path = 'results/data/Experiment-%s' % (experiment_number)
        plot_results(args.G, args.nruns, avg_fitness_vals, best_fitness_vals, num_correct_bits,
                     args.autoscale, plot_file_path, experiment_number, args.ce)

if __name__ == "__main__":
    main()
