#!/usr/bin/env python
# -*- coding: utf-8 -*-
##############################################################
# ffs.py, fitness functions to be used in
#         genetic_algorithms.py
#
# Written by Jared Smith for COSC 427/527
# at the University of Tennessee, Knoxville.
###############################################################


def fitness_func_1(bit_sum, l):
    return (pow(((bit_sum / pow(2, l)), 10)))


def fitness_func_2(bit_sum, l):
    return (pow((((1 - bit_sum) / pow(2, l)), 10)))
