#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################
# genetic_algorithms.py, general purpose genetic
#                        algorithm in Python
#
# Written by Jared Smith and David Cunningham for COSC 427/527
# at the University of Tennessee, Knoxville.
###############################################################
# TODO:
# - Generalize ff call to fitness function using kwargs argument
#
###############################################################

import random

import bitstring as bs
from scipy import stats
import numpy as np

import ffs


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
            current_offspring is the current generation of children
            nruns is the number of runs to run the algorithm
            learn is a bool specifying whether to learn the offspring
            NG is the number of guesses to use when learning the offspring
            ff is the fitness function to use
            ce is a bool specifying whether to inflict a sudden change of
            environment on the final population

    """

    def __init__(self, args, logger):
        # Parameters of the algorithm
        self.args = args
        self.l = args.l
        self.N = args.N
        self.G = args.G
        self.pr_mutation = args.pm
        self.pr_crossover = args.pc
        self.population = []
        self.current_offspring = []
        self.nruns = args.nruns
        self.NG = args.NG
        self.learn = args.learn
        self.ff = args.ff
        self.ce = args.ce
        self.max_recovery = 100

        # Helper objects
        self.logger = logger
        self.orig_fitness_vals = np.zeros((self.G + self.max_recovery, self.N))
        self.norm_fitness_vals = np.zeros((self.G + self.max_recovery, self.N))
        self.total_fitness_vals = np.zeros((self.G + self.max_recovery, self.N))
        self.parents = [[None, None] for x in xrange(self.G + self.max_recovery)]
        self.pr_mut_dist = None
        self.pr_cr_dist = None
        self.env_state = 0

        # Statistics Objects
        self.avg_fitness_vals = np.zeros((self.nruns, 2,  self.G + self.max_recovery))
        self.best_fitness_vals = np.zeros((self.nruns, 2,  self.G + self.max_recovery))
        self.num_correct_bits = np.zeros((self.nruns, 2, self.G + self.max_recovery))
        self.recovery_time = 0


    def initialize_algorithm(self):
        # Initialize the population
        self.logger.info("Initializing population...")
        self.initialize_pop()
        self.logger.info("Initialization Complete.")

        # Generate probablility distributions for mutation and crossover
        self.logger.info("Generating probability distributions for mutation and " +
                         "crossover...")
        self.generate_prob_distributions()
        self.logger.info("Generated probability distributions for mutation and " +
                         "crossover.")

    # Initialize the Population
    def initialize_pop(self):
        # Generate N random bitstrings
        self.population = []
        self.current_offspring = []
        for i in range(self.N):
            tmp_bitstring = ''.join(random.choice('01') for _ in range(self.N))
            tmp_bitstring = bs.BitArray(bin=tmp_bitstring)
            self.population.append(tmp_bitstring)

    # Initialize the genetic environment
    def initialize_env(self):
        # Get the appropriate fitness function
        self.logger.info("Initializing environment...")
        if self.env_state != 0:
            self.recovery_time = 0
        self.logger.info("Initialized environment.")

    # Generate probability distributions for mutation and crossover
    def generate_prob_distributions(self):
        # xk is an array of size 2 (0, 1), that represents the possible
        # values we can get out of the distribution
        xk = np.arange(2)

        # pk1 and pk2 are the probabilities for getting the corresponding
        # xk value for mutation and crossover, respectively
        pk1 = (1 - self.pr_mutation, self.pr_mutation)
        pk2 = (1 - self.pr_crossover, self.pr_crossover)

        # Generate the object that will be used to get random numbers
        # according to each distribution.
        self.pr_mut_dist = stats.rv_discrete(name='pr_mut_dist',
                                                values=(xk, pk1))
        self.pr_cr_dist = stats.rv_discrete(name='pr_cr_dist',
                                               values=(xk, pk2))

    # Calculate the fitness of each individual of the population
    def fitness(self, g, nrun, ff=ffs.fitness_func_1):
        total_fitness = 0

        self.logger.debug("Getting fitness of generation %d" % g)

        # Step through each bitstring in the population
        for i, bitstring in enumerate(self.population):
            # Get the integer value of the string
            bit_sum = bitstring.uint
            self.logger.debug("Sum of bitstring at %d of population: %d" % (i, bit_sum))
            fitness_val = self.ff(bit_sum, self.l)
            self.orig_fitness_vals[g][i] = fitness_val
            self.logger.debug("Fitness Value at index %d in population: %lf" % (i, fitness_val))
            total_fitness += fitness_val

        self.logger.debug("Total fitness from step 1: %lf" % total_fitness)

        self.norm_fitness_vals[g] = self.orig_fitness_vals[g] / total_fitness
        self.logger.debug("Sum of norm fitness vals: %lf" % np.sum(self.norm_fitness_vals[g]))

        prev_norm_fitness_val = 0
        for i in range(self.N):
            self.logger.debug("Normalized Fitness Value at index %d in population: %lf" % (i, self.norm_fitness_vals[g][i]))
            self.total_fitness_vals[g][i] = (
                self.norm_fitness_vals[g][i] + prev_norm_fitness_val)
            prev_norm_fitness_val = self.total_fitness_vals[g][i]
            self.logger.debug("Total Fitness Value at index %d in population: %lf" % (i, self.total_fitness_vals[g][i]))

    # Select parents from population
    def select(self, g):
            rand_nums = np.random.uniform(0, 1, 2)

            # Select the first parent
            self.logger.info("Selecting the first parent...")
            prev_individual_fit = 0
            j = 0
            while True:
                if j >= self.N:
                    j = 0
                    rand_nums = np.random.uniform(0, 1, 2)

                individual_fit = self.total_fitness_vals[g][j]
                if rand_nums[0] < self.total_fitness_vals[g][0]:
                    self.logger.debug("1: Prev_Individual Fit: %lf" % prev_individual_fit)
                    self.logger.debug("1: Individual Fit: %lf" % individual_fit)
                    self.logger.debug("1: Rand Num: %lf" % rand_nums[0])
                    self.logger.debug("1: Population at %d: %s" % (j, self.population[j].bin))
                    self.parents[g][0] = self.population[0]
                    break

                if j != 0:
                    self.logger.debug("1: Prev_Individual Fit: %lf" % prev_individual_fit)
                    self.logger.debug("1: Individual Fit: %lf" % individual_fit)
                    self.logger.debug("1: Rand Num: %lf" % rand_nums[0])
                    self.logger.debug("1: Population at %d: %s" % (j, self.population[j].bin))

                    if (prev_individual_fit <= rand_nums[0] <= individual_fit):
                        self.logger.debug("1: selected individuval from population at %d: %s" % (j, self.population[j].bin))
                        self.parents[g][0] = self.population[j]
                        self.logger.debug("1: parents[%d][0]: %s" % (g, str(self.parents[g][0])))
                        break
                prev_individual_fit = individual_fit
                j += 1
            self.logger.info("First parent has been selected.")

            # Select the second parent
            self.logger.info("Selecting the second parent...")
            prev_individual_fit = 0
            j = 0
            cycles = 0
            while True:
                if j >= self.N:
                    cycles += j
                    if cycles >= 100:
                        self.parents[g][1] = self.parents[g][0]
                        break
                    else:
                        j = 0
                        rand_nums = np.random.uniform(0, 1, 2)

                individual_fit = self.total_fitness_vals[g][j]
                if rand_nums[1] < self.total_fitness_vals[g][0]:
                    self.logger.debug("2: prev_individual fit: %lf" % prev_individual_fit)
                    self.logger.debug("2: individual fit: %lf" % individual_fit)
                    self.logger.debug("2: rand num: %lf" % rand_nums[1])
                    self.logger.debug("2: population at %d: %s" % (j, self.population[j].bin))
                    self.parents[g][1] = self.population[0]
                    break

                if j != 0:
                    self.logger.debug("2: prev_individual fit: %lf" % prev_individual_fit)
                    self.logger.debug("2: individual fit: %lf" % individual_fit)
                    self.logger.debug("2: rand num: %lf" % rand_nums[1])
                    self.logger.debug("2: population at %d: %s" % (j, self.population[j].bin))

                    if (prev_individual_fit <= rand_nums[1] <= individual_fit):
                        if (self.population[j] != self.parents[g][0]):
                            self.logger.debug("2: selected individuval from population at %d: %s" % (j, self.population[j].bin))
                            self.parents[g][1] = self.population[j]
                            self.logger.debug("1: parents[%d][0]: %s" % (g, str(self.parents[g][1])))
                            break

                prev_individual_fit = individual_fit
                j += 1

            self.logger.info("Second parent has been selected.")

    # Mutate the parents
    def mutate(self, g):

        for parent in self.parents[g]:
            for index_bit in xrange(0, self.N):
                # Determine whether we will perform a mutation on the bit
                to_mutate = self.pr_mut_dist.rvs(size=1)

                # Mutate the bit if choice is 1, otherwise skip it
                if to_mutate:
                    parent.invert(index_bit)


    # Crossover the parents
    def crossover(self, g):
        to_crossover = self.pr_cr_dist.rvs(size=1)

        # Crossover the parents if to_crossover is 1, otherwise copy the
        # parents exactly into the children
        if to_crossover:
            # Create empty children
            c1 = bs.BitArray(length=self.N)
            c2 = bs.BitArray(length=self.N)

            # Select the bit at which to crossover the parents
            crossover_bit = random.randint(0, self.l)

            # Perform the crossover
            c1.overwrite(self.parents[g][0][:crossover_bit], 0)
            c1.overwrite(self.parents[g][1][:(self.l - crossover_bit)], crossover_bit)
            c2.overwrite(self.parents[g][1][:crossover_bit], 0)
            c1.overwrite(self.parents[g][0][:(self.l - crossover_bit)], crossover_bit)

            self.current_offspring.append(c1)
            self.current_offspring.append(c2)
        else:
            self.current_offspring.append(self.parents[g][0])
            self.current_offspring.append(self.parents[g][1])

    # Learn the children on the fitness function.
    def learn_offspring(self, g, ff=ffs.fitness_func_1):
        # For every child in the current generation, iterate for NG guesses,
        # manipulating the child every time trying to find a best fitness,
        # and when the children are exhausted, the children will have been
        # fine tuned according to the original fitness function.
        for child in self.current_offspring:
            for guess in xrange(0, self.NG):
                current_child = child.copy()
                max_fitness = 0

                for ibit in xrange(0, self.l):
                    if random.choice([0, 1]):
                        if current_child[ibit]:
                            current_child.set(False, ibit)
                        elif not current_child[ibit]:
                            current_child.set(True, ibit)

                bit_sum = current_child.uint
                current_fitness = ff(bit_sum, self.l)
                max_fitness = max(current_fitness, max_fitness)

                if current_fitness == max_fitness:
                    child = current_child

    def compute_statistics(self, g, nrun):
        # Get the number of correct bits in the best individual
        index_bi = self.orig_fitness_vals[g].argmax()
        bi_bitstring = self.population[index_bi]
        individual_num_correct_bits = bi_bitstring.count(1)
        self.num_correct_bits[nrun][self.env_state][g] = individual_num_correct_bits

        # Get the numerical value of the best fitness
        self.best_fitness_vals[nrun][self.env_state][g] = self.orig_fitness_vals[g][index_bi]

        # Get the average value of the fitness
        self.avg_fitness_vals[nrun][self.env_state][g] = np.average(self.orig_fitness_vals[g])

        # Logging computed statistics to stdout and to file
        self.logger.info("Number of Correct Bits in Best Individual: %d"
                         % individual_num_correct_bits)
        self.logger.info("Fitness Value of Best Individual: %lf"
                         % self.best_fitness_vals[nrun][self.env_state][g])
        self.logger.info("Average Fitness Value of Generation: %lf"
                         % self.avg_fitness_vals[nrun][self.env_state][g])

    # Check if the population has recovered from an environment change
    def check_population_recovery(self, g, nrun):
        checks = []
        checks.append((self.best_fitness_vals[nrun][1][g] > self.best_fitness_vals[nrun][0][self.G - 1]))
        checks.append((self.avg_fitness_vals[nrun][1][g] > self.avg_fitness_vals[nrun][0][self.G - 1]))
        for check in checks:
            print check
        if all(checks):
            return True

    # Run one generation
    def reproduce(self, nrun, g):
        self.logger.info("Running fitness function on generation " +
                            "%d..." % g)
        self.fitness(g, nrun, self.ff)

        # Select the parents of the next generation and generate the
        # new offspring.
        for i in range(self.N / 2):
            self.logger.info("Selecting the parents of generation %d..."
                                % g)
            self.select(g)
            self.logger.info("Selection of the parents of generation " +
                                "%d finished." % g)

            self.logger.info("Performing crossover and mutation of " +
                                "the parent's offspring from generation " +
                                "%d..." % g)
            self.crossover(g)
            self.mutate(g)
            self.logger.info("Crossover and mutation of the " +
                                "the parent's offspring of " +
                                " generation %d finished.", g)

        # Learn the offspring if specified
        if self.learn:
            self.logger.info("Performing learning on the offspring" +
                                " of generation %d..." % g)
            self.learn_offspring(g, self.ff)
            self.logger.info("Learning on the offspring" +
                                " of generation %d finished." % g)

        # Compute statistics for this generation
        self.logger.info("Computing statistics for Run %d, Generation %d..." % (nrun, g))
        self.compute_statistics(g, nrun)
        self.logger.info("Computation of statistics finished for Run %d, Generation %d." % (nrun, g))

        # Replace the old population with the new population
        self.population = self.current_offspring
        self.current_offspring = []

    # Run through the total runs specified
    def run(self):
        for nrun in range(self.nruns):
            self.logger.info("Starting run %d..." % nrun)
            self.initialize_algorithm()

            self.env_state = 0
            self.ff = ffs.fitness_func_1
            self.logger.info("Fitness Function before normal generation run: %s" % str(self.ff.__name__))

            for g in range(self.G):
                self.logger.info("Generation %d running..." % g)
                self.reproduce(nrun, g)
                self.logger.info("Generation %d finished." % g)

            if self.ce:
                self.logger.info("Running Sudden change in environment test starting at generation %d..." % g)
                self.env_state = 1
                self.initialize_env()
                self.ff = ffs.fitness_func_2
                self.logger.info("Fitness Function before changed environment generation run: %s" % str(self.ff.__name__))

                while True:
                    g += 1
                    self.recovery_time += 1
                    self.logger.info("Recovery Time: %d, Max Recovery: %d, Generation %d In Changed Environment" % (self.recovery_time, self.max_recovery, g))
                    if self.recovery_time >= self.max_recovery:
                        self.logger.info("Population has not recovered after %d iterations. Quitting now." % self.recovery_time)
                        break
                    self.reproduce(nrun, g)
                    if self.check_population_recovery(g, nrun):
                        self.logger.info("Population has recovered after %d iterations." % self.recovery_time)
                        break
                    self.logger.info("Population has not recovered...continuing generation.")

            self.logger.info("Finished run %d." % nrun)

        return (self.avg_fitness_vals, self.best_fitness_vals,
                self.num_correct_bits)
