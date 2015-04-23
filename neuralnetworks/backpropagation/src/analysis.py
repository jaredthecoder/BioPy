import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk 

from utils import *

def plot_cost_versus_epochs_colormap(plot_file_name, target_test, Y_pred, cost_list, cost_test_list, learning_rates):

    """ cost_test_list --> target testing error list for each epoch, where cost_test_list[i] is the testing error for epoch i.
        cost_list --> training error list for each epoch, where cost_list[i] is the training error for epoch i.
    """

    x = np.arange(len(cost_list))
    y = cost_list
    color_metric = cost_test_list

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    cmhot = plt.cm.get_cmap("hot")
    l = ax.scatter(x, y, c=color_metric, cmap=cmhot)
    fig.colorbar(l)
    plt.show()

    # Save the figure
    plt.savefig(plot_file_name, bbox_inches='tight')

def plot_cost_versus_epochs(autoscale, plot_file_path, experiment_number, cost_list, cost_test_list):

    x1 = np.arange(len(cost_list))
    y1 = cost_list

    x2 = np.arange(len(cost_test_list))
    y2 = cost_test_list

    fig = plt.figure()
    subplt = plt.subplot(111)
    plt.plot(x1, y1)
    plt.xlabel('Epochs')
    plt.ylabel('Cost Function')
    plt.title('Cost Function Per Epoch Over %s Epochs' % str(len(x1)))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True, True, True)
    fig.tight_layout()  

    plot_file_name = "%s/epoch-vs-cost-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    fig = plt.figure()
    subplt = plt.subplot(111)
    plt.plot(x2, y2)
    plt.xlabel('Epochs')
    plt.ylabel('Cost Function')
    plt.title('Cost Function Per Testing Epoch Over %s Epochs' % str(len(x2)))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True,True,True)
    fig.tight_layout()  

    plot_file_name = "%s/epoch-vs-testing-cost-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    return plot_file_name

def plot_rmse_versus_epochs(autoscale, plot_file_path, experiment_number, rmse):

    x1 = np.arange(len(rmse))
    y1 = rmse

    fig = plt.figure()
    subplt = plt.subplot(111)
    plt.plot(x1, y1)
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('RMSE Per Epoch Over %s Epochs' % str(len(x1)))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True,True,True)
    fig.tight_layout()  

    plot_file_name = "%s/epoch-vs-rmse-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    return plot_file_name

def plot_learning_rates_versus_epochs(autoscale, plot_file_path, experiment_number, learning_rates):
    x1 = np.arange(len(learning_rates))
    y1 = learning_rates

    fig = plt.figure()
    subplt = plt.subplot(111)
    plt.plot(x1, y1)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Per Epoch Over %s Epochs' % str(len(x1)))
    plt.grid()
    if autoscale:
        subplt.autoscale_view(True,True,True)
    fig.tight_layout()  

    plot_file_name = "%s/epoch-vs-lr-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    return plot_file_name

def plot_accuracy(plot_file_path, experiment_number, target_test, Y_pred):

    x1 = target_test
    y1 = Y_pred

    fig = plt.figure()
    plt.scatter(x1, y1, alpha=0.5)
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.title('Accuracy of Network')
    plt.grid()
    fig.tight_layout()

    plot_file_name = "%s/accuracy-exp-%s.pdf" % (plot_file_path, experiment_number)
    plt.savefig(plot_file_name, bbox_inches='tight')

    return plot_file_name

def facial_recognition_graphs():
    
    prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, h, w)

    # plot the gallery of the most significative eigenfaces

    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
    plot_gallery(eigenfaces, eigenface_titles, h, w)

    plt.show()