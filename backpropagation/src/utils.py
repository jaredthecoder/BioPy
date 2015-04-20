import numpy as np

def sigmoid(z):
    """sigmoid is a basic sigmoid function returning values from 0-1"""
    return 1.0 / ( 1.0 + np.exp(-z) )

def sigmoidGradient(z):
    # Not used
    return self.sigmoid(z) * ( 1 - self.sigmoid(z) )

def format__1(digits,num):
        if digits<len(str(num)):
            raise Exception("digits<len(str(num))")
        return ' '*(digits-len(str(num))) + str(num)

def printmat(arr,row_labels=[], col_labels=[]): #print a 2d numpy array (maybe) or nested list
    max_chars = max([len(str(item)) for item in flattenList(arr)+col_labels]) #the maximum number of chars required to display any item in list
    if row_labels==[] and col_labels==[]:
        for row in arr:
            print '[%s]' %(' '.join(format__1(max_chars,i) for i in row))
    elif row_labels!=[] and col_labels!=[]:
        rw = max([len(str(item)) for item in row_labels]) #max char width of row__labels
        print '%s %s' % (' '*(rw+1), ' '.join(format__1(max_chars,i) for i in col_labels))
        for row_label, row in zip(row_labels, arr):
            print '%s [%s]' % (format__1(rw,row_label), ' '.join(format__1(max_chars,i) for i in row))
    else:
        raise Exception("This case is not implemented...either both row_labels and col_labels must be given or neither.")

def save_data(save_path, target_test, Y_pred, cost_list, cost_test_list, learning_rates, rmse, experiment_number):
    
    # Saving error/cost values

    # Save in .npz, which is easily readable by Python Numpy for later use
    cost_npz_file = "%s/cost-info-file-npz-exp-%s.npz" % (save_path, experiment_number)
    np.savez(cost_npz_file, cost_list, cost_test_list)

    # Also, save as text for human readability
    cost_txt_file = "%s/cost-info-file-txt-exp-%s.txt" % (save_path, experiment_number)
    np.savetxt(cost_txt_file, cost_list, fmt='%.8f', delimiter=',')

    cost_test_txt_file = "%s/cost-test-info-file-txt-exp-%s.txt" % (save_path, experiment_number)
    np.savetxt(cost_test_txt_file, cost_test_list, fmt='%.8f', delimiter=',')


    # Saving target and predicted values values

    # Save in .npz, which is easily readable by Python Numpy for later use
    tp_npz_file = "%s/target-predicted-info-file-npz-exp-%s.npz" % (save_path, experiment_number)
    np.savez(tp_npz_file, target_test, Y_pred)

    # Also, save as text for human readability
    target_txt_file = "%s/target-info-file-txt-exp-%s.txt" % (save_path, experiment_number)
    np.savetxt(target_txt_file, target_test, fmt='%.8f', delimiter=',')

    predicted_txt_file = "%s/predicted-info-file-txt-exp-%s.txt" % (save_path, experiment_number)
    np.savetxt(predicted_txt_file, Y_pred, fmt='%.8f', delimiter=',')


    # Save learning rates

    # Save in .npz, which is easily readable by Python Numpy for later use
    lr_npz_file = "%s/learning-rates-info-file-npz-exp-%s.npz" % (save_path, experiment_number)
    np.savez(lr_npz_file, learning_rates)

    # Also, save as text for human readability
    lr_txt_file = "%s/learning-rates-info-file-txt-exp-%s.txt" % (save_path, experiment_number)
    np.savetxt(lr_txt_file, learning_rates, fmt='%.8f', delimiter=',')

    # Save RMSE

    # Save in .npz, which is easily readable by Python Numpy for later use
    rmse_npz_file = "%s/rmse-info-file-npz-exp-%s.npz" % (save_path, experiment_number)
    np.savez(rmse_npz_file, rmse)

    # Also, save as text for human readability
    rmse_txt_file = "%s/rmse-info-file-txt-exp-%s.txt" % (save_path, experiment_number)
    np.savetxt(rmse_txt_file, rmse, fmt='%.8f', delimiter=',')

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





