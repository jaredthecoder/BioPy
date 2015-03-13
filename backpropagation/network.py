# Built-in Python Libraries
import random

# Third Party Libraries
import numpy as np

random.seed()

class InvalidWeightsException(Exception):
    pass

class InvalidNetworkInputException(Exception):
    pass

class BackPropagationNetwork(object):
    def __init__(self):

