import numpy as np
from matplotlib.pyplot import plot
from sklearn.datasets import load_iris, load_digits

from network import BackPropagationNetwork

class Tests(object):

    def __init__(self):
        pass

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

        print
        print 'Data Set: %d Observations, %d Features' % (data_array.shape[0], data_array.shape[1])
        print 'Training Set: %d Observations, %d Features' % (train_data.shape[0], train_data.shape[1])
        print 'Test Set: %d Observations, %d Features' % (test_data.shape[0], test_data.shape[1])
        print
        print 'Target Set: %d Observations, %d Classes' % (target_array.shape[0], target_array.shape[1])
        print 'Training Set: %d Observations, %d Features' % (train_target.shape[0], train_target.shape[1])
        print 'Test Set: %d Observations, %d Features' % (test_target.shape[0], test_target.shape[1])
        print

        return train_data, test_data, train_target, test_target

    def iris_test(self, hidden_unit_length_list = [], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5):
        NN = BackPropagationNetwork()

        data_set = load_iris()

        data = data_set.data
        target = self.translate_to_binary_array(data_set.target)

        # Split into train, test sets
        data_train, data_test, target_train, target_test = self.train_test_split(data, target, .75)

        return BackPropagationNetwork.nn_test(NN, data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)

    def digits_test(self, hidden_unit_length_list = [], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5):
        NN = BackPropagationNetwork()

        data_set = load_digits()

        data = data_set.data
        target = self.translate_to_binary_array(data_set.target)

        # Split into train, test sets
        data_train, data_test, target_train, target_test = self.train_test_split(data, target, .75)

        return BackPropagationNetwork.nn_test(NN, data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)


    def XOR_test(self, hidden_unit_length_list = [], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5):
        """
        XOR_test is a simple test of the nn printing the predicted value to std out
        Trains on a sample XOR data set
        Predicts a single value
        Accepts an option parameter to set architecture of hidden layers
        """

        NN = BackPropagationNetwork()

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

        print 'Training Data: X & Y'
        print data_train
        print target_train

        return BackPropagationNetwork.nn_test(NN, data_train, target_train, hidden_unit_length_list, epochs, learning_rate, momentum_rate, learning_acceleration, learning_backup, data_test, target_test)
        
