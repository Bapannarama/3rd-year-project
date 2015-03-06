__author__ = 'bapanna'

import shepard as sp


def nn_classifier(rssi, vector):
	errors = [[0 for i in range(len(rssi))] for i in range(len(rssi[0]))]

	for i in range(len(rssi)):
		for j in range(len(rssi[i])):
			# calculates how far apart vector and the coordinate readings are
			errors[i][j] = sp.euclidean_distance(vector, rssi[i][j])

	min_in_each_col = []

	for col in errors:
		min_in_each_col.append(min(col))

	x_coord = min_in_each_col.index(min(min_in_each_col))
	y_coord = errors[x_coord].index(min(errors[x_coord]))

	return [x_coord, y_coord]


def knn_regressor(rssi, vector, k):
	# function needs to return an average of the k coordinates which are nearest
	# in feature space
	errors = [[0 for i in range(len(rssi))] for i in range(len(rssi[0]))]

	for i in range(len(rssi)):
		for j in range(len(rssi[i])):
			# calculates how far apart vector and the coordinate readings are
			errors[i][j] = sp.euclidean_distance(vector, rssi[i][j])

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