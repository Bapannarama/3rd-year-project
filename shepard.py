__author__ = 'bapanna'

import numpy as np


def denominator(w):
	return np.sum(w)


def numerator(w):
	x = 0
	y = 0

	for i in range(len(w)):
		for j in range(len(w[0])):
			x += i * w[i][j]
			y += j * w[i][j]

	return [x,y]


def euclidean_distance(r, s):
	r = np.asarray(r)
	s = np.asarray(s)

	return np.linalg.norm(r - s)


def calc_weights(database, rssi_vec, p):
	w = [[0 for i in range(len(database))] for i in range(len(database[0]))]

	for i in range(len(database)):
		for j in range(len(database[i])):
			# prevents division by zero when testing with zeroed vector from tests.py
			if database[i][j][0] == 0:
				w[i][j] = 0
				continue

			data_vector = database[i][j]

			w[i][j] = euclidean_distance(data_vector, rssi_vec) ** -p

	return w


# checks if input vector corresponds to RSSIs of a known point
def check_input_vector(database, vector):
	for i, col in enumerate(database):
		for j, strengths in enumerate(col):
			if strengths == vector:
				return [i,j]

	return False


# rssi will be a list
def shepard_interpolation(rssi_data, rssi_vector, power):
	location = check_input_vector(rssi_data, rssi_vector)

	if location:
		return location

	weights = calc_weights(rssi_data, rssi_vector, power)
	num = numerator(weights)
	den = denominator(weights)

	coordinates = num / den

	return coordinates