__author__ = 'bapanna'

import data_fetch as df
import data_repr as dr
import numpy as np
import shepard as sp
import knn


def generate_errors_matrix(rssi, function, p=2, k=3):
	"""This function has 'function' representing the interpolation scheme as a
	parameter. This allows flexibility in which method is used to find position
	estimates. The rssi parameter is expected to be averaged."""

	# scheme to be used determined by value of 'function' variable

	# This function handles the looping of the interpolation function across all
	# grid points.

	errors = [[0 for x in range(len(rssi))] for x in range(len(rssi[0]))]

	for i in range(len(rssi)):
		for j in range(len(rssi[i])):
			# test_vector contains the rssi values of the coordinate to be tested
			test_vector = rssi[i][j]

			# test_database is the database which the test_vector will be tested against
			rssi[i][j] = [0, 0, 0, 0]

			test_coordinate = np.asarray((i,j))

			if function == sp.shepard_interpolation:
				calc_coordinate = function(rssi, test_vector, p)

			elif function == knn.nn_classifier:
				calc_coordinate = function(rssi, test_vector)

			elif function == knn.knn_regressor:
				calc_coordinate = knn.knn_regressor(rssi, test_vector, k)

			calc_coordinate = np.asarray(calc_coordinate)
			errors[i][j] = np.linalg.norm(test_coordinate - calc_coordinate) * 50

			rssi[i][j] = test_vector

	return errors


dataset = 'elevated'
s, t = df.fetch_data(dataset)
s = df.rssi_average(s)
func = sp.shepard_interpolation
error_matrix = generate_errors_matrix(s, func)
dr.errors_histogram_show(error_matrix)