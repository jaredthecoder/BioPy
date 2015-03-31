import numpy as np

def sigmoid(z):
    """sigmoid is a basic sigmoid function returning values from 0-1"""
    return 1.0 / ( 1.0 + np.exp(-z) )

def sigmoidGradient(z):
    # Not used
    return self.sigmoid(z) * ( 1 - self.sigmoid(z) )
