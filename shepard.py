__author__ = 'bapanna'

import numpy as np


def denominator(w):
	return np.sum(w)


def numerator(w):
	# multiplies each point by corresponding weight
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


def calc_weights(fingerprint, rssi_vector, p):
	"""
	:param fingerprint: 7*7*4 list of reference rssi values
	:param rssi_vector: list of length 4 of rssi values from each reader
	:param p: power for weight calculation
	:return: 7*7 matrix containing a weight for each point
	"""
	w = [[0 for i in range(len(fingerprint))] for i in range(len(fingerprint[0]))]

	for i in range(len(fingerprint)):
		for j in range(len(fingerprint[i])):
			# prevents division by zero when testing with zeroed vector from tests.py
			if fingerprint[i][j][0] == 0:
				w[i][j] = 0
				continue

			data_vector = fingerprint[i][j]

			w[i][j] = euclidean_distance(data_vector, rssi_vector) ** -p

	return w


# checks if input vector corresponds to RSSIs of a known point
def check_input_vector(fingerprint, rssi_vector):
	"""
	:param fingerprint: 7*7*4 list of reference rssi values
	:param rssi_vector: list of length 4 containing rssi values from each reader
	:return: boolean to check if rssi_vector is identical to any point in fingerprint
	"""
	for i, col in enumerate(fingerprint):
		for j, strengths in enumerate(col):
			if strengths == rssi_vector:
				return [i,j]

	return False


# rssi will be a list
def shepard_interpolation(fingerprint, rssi_vector, power=3):
	"""
	:param fingerprint: 7*7*4 list of reference rssi values
	:param rssi_vector: list of length 4 of rssi values from each reader
	:param power: power value to be used for weight calculation step
	:return: xy coordinate pair (list) of interpolated position
	"""
	location = check_input_vector(fingerprint, rssi_vector)

	if location:
		return location

	weights = calc_weights(fingerprint, rssi_vector, power)
	num = numerator(weights)
	den = denominator(weights)

	coordinates = num / den

	return coordinates