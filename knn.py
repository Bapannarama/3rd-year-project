__author__ = 'bapanna'

import shepard as sp
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def nn_interpolation(rssi, vector):
	errors = [[0 for i in range(len(rssi))] for i in range(len(rssi[0]))]

	for i in range(len(rssi)):
		for j in range(len(rssi[i])):
			# calculates how far apart the values are
			errors[i][j] = sp.euclidean_distance(vector, rssi[i][j])

	min_error_indices_cols = []

	# stores the index of the minimum value of errors for each COLUMN from the errors list
	for i in range(len(errors)):
		min_error_indices_cols.append(np.argmin(errors[i]))

	min_errors_across_x = []
	for i, error_index in enumerate(min_error_indices_cols):
		min_errors_across_x.append(errors[i][error_index])

	min_error_x_index = np.argmin(min_errors_across_x)

	return [min_error_x_index, min_error_indices_cols[min_error_x_index]]


def knn_interpolation():
	pass