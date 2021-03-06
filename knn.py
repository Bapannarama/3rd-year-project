__author__ = 'bapanna'

import shepard as sp


def nn_classifier(fingerprint, rssi_vector):
	"""
	:param fingerprint: 7*7*4 list of reference rssi data
	:param rssi_vector: list of length 4 for a particular point with rssi
	readings from each reader
	:return: an xy coordinate pair (list) of the most similar point in the grid
	"""

	errors = [[0 for i in range(len(fingerprint))] for i in range(len(fingerprint[0]))]

	for i in range(len(fingerprint)):
		for j in range(len(fingerprint[i])):
			# calculates how far apart rssi_vector and the coordinate readings are
			errors[i][j] = sp.euclidean_distance(rssi_vector, fingerprint[i][j])

	min_in_each_col = []

	for col in errors:
		min_in_each_col.append(min(col))

	x_coord = min_in_each_col.index(min(min_in_each_col))
	y_coord = errors[x_coord].index(min(errors[x_coord]))

	return [x_coord, y_coord]


def knn_regressor(fingerprint, rssi_vector, k=11):
	"""
	This function takes in the fingerprint, a value for k and and an rssi vector
	of length 4 (each value is the value from each reader)
	"""
	# function needs to return an average of the k coordinates which are nearest
	# in feature space
	errors = [[0 for i in range(len(fingerprint))] for i in range(len(fingerprint[0]))]

	for i in range(len(fingerprint)):
		for j in range(len(fingerprint[i])):
			# calculates how far apart rssi_vector and the coordinate readings are
			errors[i][j] = sp.euclidean_distance(rssi_vector, fingerprint[i][j])

	k_nearest = []

	for r in range(k):
		min_in_each_col = []

		# return minimum error in each column
		for col in errors:
			min_in_each_col.append(min(col))

		# x coordinate of the minimum error in the errors matrix
		x_coor = min_in_each_col.index(min(min_in_each_col))
		y_coor = errors[x_coor].index(min(errors[x_coor]))

		k_nearest.append([x_coor, y_coor])

		# set current to minimum to 100 in order to find next lowest error on
		# next iteration
		errors[x_coor][y_coor] = 100

	# following lines find mean coordinate
	sum_coor = [0, 0]

	for c in k_nearest:
		sum_coor[0] += c[0]
		sum_coor[1] += c[1]

	result = [sum_coor[0] / len(k_nearest), sum_coor[1] / len(k_nearest)]
	return result