import numpy as np


def min(matrix, n=1):
    # Get the first matrix of minimum elements.
    min1 = np.amin(matrix, axis=0)
    # Get the first matrix of maximum elements.
    max1 = np.amax(matrix, axis=0)
    for i in range(n-1):
        # Create a mask (a matrix with True or False). If True is evaluable for
        # the second min.
        mask = matrix > min1
        # Use the mask to if greater than m mantain the value of m(i,j), else
        # use the max1(i,j) element.
        matrix2 = np.where(mask, matrix, max1)
        # Get the second matrix of minimum elements.
        min2 = np.amin(matrix2, axis=0)
        # Prepare matrix and min1 for the next_round
        matrix = matrix2
        min1 = min2
    return min1

m = np.random.rand(3, 4, 5)
print "Shape: ", m.shape
print "Matrix: \n", m
print "First min: \n", min(m, n=1)
print "Second min: \n", min(m, n=2)
print "Third min: \n", min(m, n=3)
