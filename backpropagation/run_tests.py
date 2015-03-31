from network import BackPropagationNetwork
from tests import Tests

if __name__=="__main__":
    print "====RUNNING TESTS===="

    print "\n"
    print "====XOR TEST===="
    nn_xor = Tests()

    target_test, Y_pred, J_list, J_test_list, learning_rates = nn_xor.XOR_test(hidden_unit_length_list = [2], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5)

    print "\n"
    print "====DIGITS TEST===="
    nn_digits = Tests()

    target_test, Y_pred, J_list, J_test_list, learning_rates = nn_digits.digits_test(hidden_unit_length_list = [2], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5)

    print "\n"
    print "====IRIS TEST===="
    nn_iris = Tests()

    target_test, Y_pred, J_list, J_test_list, learning_rates = nn_iris.iris_test(hidden_unit_length_list = [2], epochs=2500, learning_rate=0.5, momentum_rate=0.1, learning_acceleration=1.05, learning_backup=0.5)
