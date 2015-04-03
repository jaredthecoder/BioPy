from sklearn.datasets import load_iris, load_digits, fetch_lfw_people, fetch_lfw_pairs
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import train_test_split as sklearn_train_test_split
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_rates_versus_epochs(num_image, autoscale, learning_rates, plot_file_path=None):
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

    plt.show()

    if plot_file_path is None:
        plot_file_name = "learning_rates-faces-%s.pdf" % (str(num_image))
    else:
        plot_file_name = "%s/learning_rates-faces-%s.pdf" % (plot_file_path, str(num_image))

    plt.savefig(plot_file_name, bbox_inches='tight')


def plot_gallery(num_image, images, titles, h, w, n_row=3, n_col=4, plot_file_path=None):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

    if plot_file_path is None:
        plot_file_name = "gallery-image-%s.pdf" % (num_image)
    else:
        plot_file_name = "%s/gallery-image-%s.pdf" % (plot_file_path, num_image)

    plt.savefig(plot_file_name, bbox_inches='tight')

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
# y = self.translate_to_binary_array(y)

target_names = lfw_people.target_names
n_classes = target_names.shape[0]

# split into a training and testing set
X_train, X_test, y_train, y_test = sklearn_train_test_split(
    X, y, test_size=0.25)

y_pred = None
y_test = None
with np.load('target-predicted-info-file-npz-exp-1.npz') as data:
    y_pred = data['arr_1']
    y_test = data['arr_0']

learning_rates = None
with np.load('learning-rates-info-file-npz-exp-1.npz') as data:
    learning_rates = data['arr_0']

plot_learning_rates_versus_epochs(1, False, learning_rates)


prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(1, X_test, prediction_titles, h, w)
