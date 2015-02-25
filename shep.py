__author__ = 'bapanna'

import numpy as np
import math

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

def load_dummy_data():
	dummy_rssi = []

	reader_results = [[0 for i in range(7)] for i in range(7)]

	for i in range(7):
		for j in range(7):
			reader_results[i][j] = math.sqrt(i**2 + j**2)

	dummy_rssi.append(reader_results)
	reader_results = [[0 for i in range(7)] for i in range(7)]

	for i in range(6, -1, -1):
		for j in range(7):
			reader_results[6-i][j] = math.sqrt(i**2 + j**2)

	dummy_rssi.append(reader_results)
	reader_results = [[0 for i in range(7)] for i in range(7)]

	for i in range(7):
		for j in range(6, -1, -1):
			reader_results[i][6-j] = math.sqrt(i**2 + j**2)

	dummy_rssi.append(reader_results)
	reader_results = [[0 for i in range(7)] for i in range(7)]

	for i in range(6, -1, -1):
		for j in range(6, -1, -1):
			reader_results[6-i][6-j] = math.sqrt(i**2 + j**2)

	dummy_rssi.append(reader_results)

	return dummy_rssi

def euclidean_distance(r, s):
	r = np.asarray(r)
	s = np.asarray(s)

	return np.linalg.norm(r - s)

def calc_weights(database, rssi_vec, p):
	w = [[0 for i in range(len(database[0]))] for i in range(len(database[0][0]))]
	data_vector = []

	for x in range(len(database[0])):
		for y in range(len(database[0][0])):
			if database[0][x][y] == 0:
				w[x][y] = 0
				continue

			for i in range(len(database)):
				data_vector.append(database[i][x][y])

			w[x][y] = euclidean_distance(data_vector, rssi_vec) ** -p
			data_vector = []

	return w

# checks if input vector corresponds to RSSIs of a known point
def check_input_vector(original_database, vector):
	database = np.asarray(original_database)

	for i, x in enumerate(database[0]):
		for j, y in enumerate(x):
			if vector[0] == y:
				if vector[1] == database[1][i][j]:
					if vector[2] == database[2][i][j]:
						if vector[3] == database[3][i][j]:
							return [i,j]

	return False

# rssi will be a list
def shepard_interpolation(ref_points_rssi, rssi_vector, power):
	location = check_input_vector(ref_points_rssi, rssi_vector)

	if location:
		return location

	weights = calc_weights(ref_points_rssi, rssi_vector, power)
	num = numerator(weights)
	den = denominator(weights)

	if den == 0:
		return [-1,-1]

	else:
		coordinates = num / den

	return coordinates