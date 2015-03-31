'''
    utils.py
    PyNet, Backpropagation Neural Network class

    Written by: Jared Smith and David Cunningham
'''


# 3rd-Party Libraries
import numpy as np


# Simple Logistic Sigmoid Activation Function
def logistic(x):
    return 1.0 / (1 + np.exp(-x))

# Derivative of Simple Logistic Sigmoid
def logistic_prime(x):
    ex = np.exp(-x)
    return ex / (1 + ex)**2

# Identity Function
def identity(x):
    return x

# Derivative of Identity Function
def identity_prime(x):
    return 1

