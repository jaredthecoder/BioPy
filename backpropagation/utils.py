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
