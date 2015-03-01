__author__ = 'bapanna'

import shep
import data_representation
import numpy as np
import matplotlib.pyplot as pp

def dummy_test(rssi_vector):
	dummy_data = shep.load_dummy_data()
	coords = shep.shepard_interpolation(dummy_data, rssi_vector, 2)
	print(coords)


def generate_errors_matrix():
	"""This  generate errors matrix function handles within it the loading of
	the data, shepard interpolation operations, and the generation of the error
	matrix"""
	strengths = data_representation.setup_elevated()

	test_database = strengths
	errors = [[0 for x in range(len(strengths[0]))] for x in range(len(strengths[0][0]))]
	rssi_vector = []

	for x in range(len(strengths[0])):
		for y in range(len(strengths[0][0])):
			for r in range(len(strengths)):
				rssi_vector.append(strengths[r][x][y])
				test_database[r][x][y] = 0

			test_coordinate = np.asarray((x,y))
			calc_coordinate = shep.shepard_interpolation(test_database, rssi_vector, 2)

			if calc_coordinate[0] == -1:
				errors[x][y] = -1

			else:
				calc_coordinate = np.asarray(calc_coordinate)
				errors[x][y] = np.linalg.norm(test_coordinate - calc_coordinate) * 50

			test_database = strengths
			rssi_vector = []

	return errors

errors = generate_errors_matrix()
data_representation.generate_errors_histogram(errors)