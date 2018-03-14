import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import matplotlib.pyplot as plt


def Hungerian(y,z):
    """
    Assert y and z have same shape. The cost matrix will be computed in this function.
    How to compute cost matrix:
        Use 2 for loops and calculate l2 distance from each z to each y.
    resulting cost matrix should be a square with dims (batch_size,batch_size) when y,z have dimensions (batch_size,d)
    :param y: target noise on unit sphere
    :param z: noise estimated from images(output of neural network)
    :return: y reordered such that z's will have corresponding y's
    """
    len_y = len(y)
    len_z = len(z)
    shape_y = y.shape
    if not len_y == len_z:
        raise AssertionError
    else:
        cost_matrix = np.zeros(shape = [len_y,len_y])
        for y_index, y_vect in enumerate(y):
            for z_index, z_vect in enumerate(z):
                cost_matrix[z_index][y_index] = distance.euclidean(y_vect,z_vect)
        res = linear_sum_assignment(cost_matrix)
        new_indices = res[1]
        new_y = []
        for index in new_indices:
            new_y.append(y[index])
        return np.array(new_y).reshape(shape_y)

