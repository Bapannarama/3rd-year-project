__author__ = 'bapanna'

import data_representation
import numpy as np
import shep

# database must for formatted as database[i][j][r]
def nearest_neighbour(database, vector):
	errors = [[0 for i in range(len(database))] for j in range(len(database[0]))]

	# 'errors' matrix is buggered when running nearest neighbour
	for i in range(len(database)):
		for j in range(len(database[0])):
			errors[i][j] = shep.euclidean_distance(vector, database[i][j])

	min_error_indices_cols = []

	# stores the index of the minimum value of errors for each COLUMN from the errors list
	for i in range(len(errors)):
		min_error_indices_cols.append(np.argmin(errors[i]))

	min_errors_across_x = []
	for i, error_index in enumerate(min_error_indices_cols):
		min_errors_across_x.append(errors[i][error_index])

	min_error_x_index = np.argmin(min_errors_across_x)

	return [min_error_x_index, min_error_indices_cols[min_error_x_index]]

# this function will carry out 1NN on the entire database including the test vector
def dummy_test():
	strengths = data_representation.setup_elevated()

	#database must be reformatted
	rssi_database = [[0 for i in range(len(strengths[0]))] for i in range(len(strengths[0][0]))]
	rssi_vector = []
	for i in range(len(strengths[0])):
		for j in range(len(strengths[0][0])):
			for r in range(len(strengths)):
				rssi_vector.append(strengths[r][i][j])

			rssi_database[i][j] = rssi_vector
			rssi_vector = []

	# predict where coordinate input (rssi_database[i][j]) will be based on the training data from rssi_database
	predicted_coordinates = [[0 for i in range(len(rssi_database))] for i in range(len(rssi_database[0]))]
	for i in range(len(rssi_database)):
		for j in range(len(rssi_database[0])):
			predicted_coordinates[i][j] = nearest_neighbour(rssi_database, rssi_database[i][j])

	# return matrix of predicted coordinates at each position
	return predicted_coordinates

# this function is identical to dummy_test but removes the coordinate being tested
def nn_test():
	strengths = data_representation.setup_elevated()

	# database must be reformatted
	rssi_database = [[0 for i in range(len(strengths[0]))] for i in range(len(strengths[0][0]))]
	rssi_vector = []
	for i in range(len(strengths[0])):
		for j in range(len(strengths[0][0])):
			for r in range(len(strengths)):
				rssi_vector.append(strengths[r][i][j])

			rssi_database[i][j] = rssi_vector
			rssi_vector = []

	original_rssi_database = rssi_database

	predicted_coordinates = [[0 for i in range(len(rssi_database))] for i in range(len(rssi_database[0]))]
	for i in range(len(rssi_database)):
		for j in range(len(rssi_database[0])):
			rssi_vector = rssi_database[i][j]
			rssi_database[i][j] = [-100, -100, -100, -100]
			predicted_coordinates[i][j] = nearest_neighbour(rssi_database, rssi_vector)
			rssi_database = original_rssi_database

	return predicted_coordinates


def generate_errors_matrix(predicted_coordinates):
	"""This generate errors matrix function handles within it only the generation
	of the errors based on the predicted coordinates at each position"""
	dims = np.asarray(predicted_coordinates).shape
	errors = [[0 for i in range(dims[0])] for j in range(dims[1])]

	for i in range(dims[0]):
		for j in range(dims[1]):
			errors[i][j] = shep.euclidean_distance([i, j], predicted_coordinates[i][j]) * 50

	return errors

predictions = nn_test()
errors = generate_errors_matrix(predictions)
data_representation.generate_errors_histogram(errors)