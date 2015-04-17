__author__ = 'bapanna'

import numpy as np
import shepard as sp
import knn


def trajectory_mse(predicted_path, path_type, w=''):
	"""
	:param predicted_path: list containing the coordinates of the trajectory
	:param path_type: string defining the path which the reader took
	:param weight: alpha value
	:return: list of mse values for each point
	"""
	actual_path = []
	if path_type.lower() == "parallel":
		actual_path = [[6 - i * (0.6), 6 - i * (0.6)] for i in range(10)]
	elif path_type.lower() == "perpendicular":
		actual_path = [[i * 0.6, 6 - i * 0.6] for i in range(10)]

	mse = [(np.linalg.norm(np.array(pp) - np.array(ap)) * 50) for pp, ap in
		   zip(predicted_path, actual_path)]

	return mse


def generate_errors_matrix(fingerprint, function, p=2, k=3):
	"""This function has 'function' representing the interpolation scheme as a
	parameter. This allows flexibility in which method is used to find position
	estimates. The fingerprint parameter is expected to be averaged."""

	# scheme to be used determined by value of 'function' variable

	# This function handles the looping of the interpolation function across all
	# grid points.

	errors = [[0 for x in range(len(fingerprint))] for x in range(len(fingerprint[0]))]

	for i in range(len(fingerprint)):
		for j in range(len(fingerprint[i])):
			# test_vector contains the fingerprint values of the coordinate to be tested
			test_vector = fingerprint[i][j]

			# test_database is the database which the test_vector will be tested against
			fingerprint[i][j] = [0, 0, 0, 0]

			test_coordinate = np.asarray((i,j))

			if function == sp.shepard_interpolation:
				calc_coordinate = function(fingerprint, test_vector, p)

			elif function == knn.nn_classifier:
				calc_coordinate = function(fingerprint, test_vector)

			elif function == knn.knn_regressor:
				calc_coordinate = knn.knn_regressor(fingerprint, test_vector, k)

			calc_coordinate = np.asarray(calc_coordinate)
			errors[i][j] = np.linalg.norm(test_coordinate - calc_coordinate) * 50

			fingerprint[i][j] = test_vector

	# returns a 7*7 matrix containing prediction MSE for that position in cm
	return errors

"""
elevated best knn error: k=11 @ 130.069211329
elevated best shepard error: p=3 @ 122.511396273

reader_ground best shepard error: p=1 @ 136.177528736
reader_ground best knn error: k=49 @ 132.624796737
"""