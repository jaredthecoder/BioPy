import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk 

from utils import *

def plot_error_versus_epochs_colormap(plot_file_name, target_test, Y_pred, cost_list, cost_test_list, learning_rates):

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

def plot_error_versus_epochs(plot_file_name, target_test, Y_pred, cost_list, cost_test_list, learning_rates):

	""" cost_test_list --> target testing error list for each epoch, where cost_test_list[i] is the testing error for epoch i.
		cost_list --> training error list for each epoch, where cost_list[i] is the training error for epoch i.
	"""

	x1 = np.arange(len(cost_list))
	y1 = cost_list

	x2 = np.arange(len(cost_test_list))
	y2 = cost_test_list

	plt.subplot(111)
	plt.plot(x1, y1)
	plt.xlabel('Epochs')
	plt.ylabel('Training Error')
	plt.title('Training Error Versus Epoch Over %s Epochs' % str(len(x1)))
	plt.legend(loc=0)
	plt.grid()

	plt.subplot(111)
	plt.plot(x2, y2)
	plt.xlabel('Epochs')
	plt.ylabel('Testing Error')
	plt.title('Testing Error Versus Epoch Over %s Epochs' % str(len(x2)))
	plt.legend(loc=0)
	plt.grid()
	fig.tight_layout()	

	# Save the figure
	plt.savefig(plot_file_name, bbox_inches='tight')

	return plot_file_name