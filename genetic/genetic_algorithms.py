##############################################################
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# genetic_algorithms.py, general purpose genetic algorithm in Python
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

import bitstring as bs
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt


class BaseGeneticAlgorithm(object):



    """

        Basic class for executing a genetic algorithm.

        Parameters:
            l is the size of the bit strings (each bit is a gene)
            N is the size of the population
            G is the number of generations
            pr_mutation is the probability of mutation among the genes
            pr_crossover is the probability of crossover among the genes
            population is a list of bit strings (gene strings) of size N

   """

    def __init__(self, args, logger):
        # Parameters of the algorithm
        self.l = args.l
        self.N = args.N
        self.G = args.G
        self.pr_mutation = args.pr_mutation
        self.pr_crossover = args.pr_crossover
        self.population = []

        # Helper objects
        self.logger = logger
        self.orig_fitness_vals = numpy.zeros((G, N))
        self.norm_fitness_vals = numpy.zeros((G, N))
        self.total_fitness_vals = numpy.zeros((G, N))

        # Initialize the population
        logger.info("Initializing Population...")
        self.initialize()
        logger.info("Initialization Complete.")

    # Initialize the Population
    def initialize(self, g):
        # Generate N random bitstrings
        for i in range(self.N):
            tmp_bitstring = ''.join(random.choice('01') for _ in range(N))
            tmp_bitstring = bs.BitArray(bin=tmp_bitstring)
            self.population.append(tmp_bitstring)

    # Currently on step 3 of Van Hornweders write-up
    # Calculate the fitness of each individual of the population.
    def fitness(self):
        total_fitness = 0

        # Step through each bitstring in the population
        for i, bitstring in enumerate(self.population):
            # Get the integer value of the string
            sum = bitstring.uint
            fitness_val = (pow((sum / pow(2, self.l), 10))
            orig_fitness_vals[g][i] = fitness_val
            total_fitness += fitness_val

        for i in range(self.N):
            norm_fitness_val = (orig_fitness_vals[g][i] / total_fitness)
            norm_fitness_vals[g][i] = norm_fitness_val
            if i != 0:
                total_fitness_vals[g][i] = (norm_fitness_valsi[g][i - 1] + norm_fitness_val)
            else:
                total_fitness_vals[g][i] = norm_fitness_val

        for i in range(self.N / 2):
            rand_nums = np.random.uniform(0, 1, 2)


    # Run through all the generations
    # After the fitness function is done, this function can be finished
    def run(self):
        for g in range(self.G):
            logger.info("Running fitness function on generation %d..." % g)
            fitness(g)
            logger.info("Generation %d finished." % g)

