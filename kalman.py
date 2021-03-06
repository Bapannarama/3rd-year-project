__author__ = 'bapanna'

from filterpy import kalman
from filterpy.common import Q_discrete_white_noise
import numpy as np
import shepard as sp
import knn
import statistics

# Functions written with the help of the example written as part of the library
# documentation https://filterpy.readthedocs.org/en/latest/kalman/KalmanFilter.html


def univariate_kalman(rssi_data):
	"""The following function is used to settle on a value for each point for
	the initial radio map readings. This is intended to be a replacement or
	alternative to the fingerprint_average() function in the data_fetch module.

	**Parameters**
	rssi_data:1xn list
		is the list containing the RSSI values sent from the reader when
		taking measurements at a single coordinate."""

	# calculate the sample standard deviation of the rssi_data so for the
	# measurement noise value
	standard_deviation = statistics.stdev(rssi_data)

	# one dimension for the Kalman filter - data across time at one point is
	# being examined
	reader = kalman.KalmanFilter(dim_x=1, dim_z=1)

	# initial estimate for rssi value at point (μ)
	reader.x = np.array([[50]])

	# the state is not expected to change so STM is 1
	reader.F = np.array([[1]])

	# measurement function ensures that residual is calculated using common
	# quantities
	reader.H = np.array([[1]]) # no conversion necessary between units

	""" The covariance matrix is the property which describes how two or more
	vairables vary with each other. The main diagonal is filled with the
	variances of each of the variables, while the other positions are the
	covariance of the two quantities' axis which the value sits on. As
	covariance is commutable, the matrix will be symmetrical about the
	main diagonal """

	# covariance matrix - the initial position is unknown so the matrix is
	# initialised with a large value indicating large variance (and an almost
	# uniform distribution)
	reader.P *= 1000

	# measurement noise (aka measurement noise)
	reader.R = np.array(standard_deviation)

	# process noise - use the provided white noise generator as used in the
	# example in the documentation
	# https://filterpy.readthedocs.org/en/latest/kalman/KalmanFilter.html
	# reader.Q = Q_discrete_white_noise(dim=2, dt=3)

	# discrete noise method does not seem to work for univariate problems

	for rssi in rssi_data:
		# no control vector needed (argument u)
		reader.predict()
		reader.update(rssi)

	return reader.x


def multivariate_kalman_old(reader_data, fingerprint):
	"""
	This function will be responsible for finding the trajectory of the
	reader given the reader's returned rssi values and the matrix containing the
	area's default values.

	**Parameters**
	reader_data:1xn list
		This is the list containing the RSSI values sent from the reader as
		it moved around the grid.

	fingerprint: 7x7 list
		This is the matrix which contains standard RSSI values for each point in
		the grid. Each point should be a single value and is assumed to have
		been calculated using the fingerprint_average() or univariate_kalman()
		functions.
	"""

	# Calculate the sample standard deviation of the rssi_data so for the
	# measurement noise value.
	standard_deviation = statistics.stdev(reader_data)

	# This Kalman filter object will track the position in two dimensions using
	# only the inputted RSSI value.
	# The state variable will also store the two previous x and y values for use
	# in the state transition matrix.
	reader = kalman.KalmanFilter(dim_x=4, dim_z=1)

	# The position (state variable) is stored as cartesian coordinates and
	# initialised to the centre of the grid.
	# Feature space for this step is in terms of grid coordinates.
	# The conversion between measurement space and coordinate space is carried
	# out below.
	# List element values are: [x_0, y_0, x_-1, y_-1].
	# Talk about hackiness of adding another time period into measurements!!!!!
	reader.x = np.array([[3, 3, 3, 3]])

	"""
	The state transition matrix describes the calculation which needs to be
	carried out on the state variable in order to obtain the predicted state
	variables.

	In the book example, the state variables are the position (x) and the
	velocity (x dot) of the dog. The state transition matrix then describes the
	equations:

	x- = (x.)(dt) + x
	x. = x.

	The acceleration is constant as it is assumed to be a non-factor in this
	scenario in order to keep it simple.

	In our case, the matrix must calculate in which direction the reader is
	going and extrapolate from there. For simplicity's sake, the previous
	direction of travel could be taken as the foundation for the prediction.
	"""

	# State transition matrix - "transformation of the state matrix" must be
	# carried out.
	# This matrix first adds the difference between the current and previous
	# values for x and y, then assigns the 'current' values for x and y to the
	# 'previous' position in the output vector.
	reader.F = np.array([[2, 0, -1, 0],
						 [0, 2, 0, -1],
						 [1, 0, 0, 0],
						 [0, 1, 0, 0]])

	# Motion function not needed ('Bu' term in book).

	"""
	In the book example, a measurement function was not needed as the
	residual was in the same units as the measured quantity - the entire process
	took place in the same feature space (distance). In this case, the feature
	space is the measurement space, so RSSI. However, we cannot convert from
	position to RSSI just as we cannot convert from RSSI to position yet -
	the process is not deterministic and gains accuracy as the dataset grows
	larger.
	"""

	# Measurement function matrix. This is used to calculate the residual.
	# This is used by the object to convert from coordinates → RSSI
	reader.H = np.array()

	# Covariance/state variance matrix - this will always be of dimensions n^2
	# where n is the number of state variables. As we have no idea where the
	# reader starts and x and y are independent, they will be 0. Therefore, the
	# identity matrix to which it is initialized to can be multiplied
	# element-wise by a large number to indicate a large uncertainty.
	reader.P *= 1000

	# Measurement noise.
	reader.R = np.array(standard_deviation)

	# Process noise.
	reader.Q = Q_discrete_white_noise(dim=2, dt=3)

	positions = [[0, 0, 0, 0] for i in range(len(reader_data))]
	for i, rssi in enumerate(reader_data):
		reader.predict()
		reader.update(rssi)
		positions[i] = reader.x

	return positions


def multivariate_kalman(coordinates_matrix):
	"""
	:param coordinates_matrix: 2*10 matrix of x-y coordinates from the trajectory
	:param fingerprint: 7*7 matrix of RSSI values
	:return: 2*10 matrix of filtered coordinate pairs
	"""

	reader = kalman.KalmanFilter(dim_x=4, dim_z=2)

	# STATE VARIABLE
	# will be of format [x,y,v_x,v_y]
	# velocity is the unobserved variable
	reader.x = np.array([[3, 3, 0, 0]]).T

	# STATE TRANSITION MATRIX
	# time difference between measurements is assumed to be 3 seconds
	reader.F = np.array([[1, 0, 3, 0],
						 [0, 1, 0, 3],
						 [0, 0, 1, 0],
						 [0, 0, 0, 1]])

	# MEASUREMENT MATRIX
	"""
	The measurement function converts from state variable units. As we do not
	want to convert units until the very end of the filtering, this will not
	change x and y. The shape is derived from z = Hx.
	"""
	reader.H = np.array([[1, 0, 0, 0],
						 [0, 1, 0, 0]])

	# using sample variance
	x_var = statistics.variance([p[0] for p in coordinates_matrix])
	y_var = statistics.variance([p[1] for p in coordinates_matrix])
	# MEASUREMENT NOISE
	# assume no covariance - another naïve assumption
	reader.R = np.array([[x_var, 0],
						 [0, y_var]])

	# PROCESS NOISE - this is a complete guess
	reader.Q = np.array([[0.5, 0, 0, 0],
						 [0, 0.5, 0, 0],
						 [0, 0, 0.5, 0],
						 [0, 0, 0, 0.5]])

	# COVARIANCE MATRIX
	reader.P = np.array([[500, 0, 0, 0],
						 [0, 500, 0, 0],
						 [0, 0, 500, 0],
						 [0, 0, 0, 500]])

	positions = []
	for point in coordinates_matrix:
		reader.predict()
		reader.update(np.array(point).T)

		"""
		reader.x[0] contains the original xy coordinates passed in
		"""
		positions.append(reader.x[0])

	print(np.array(positions))
	return positions


def rssi_to_coordinates(fingerprint, trajectory, function, p=3, k=11):
	"""
	:param fingerprint: 7*7 matrix containing rssi values for each point in the grid
	:param trajectory: list of length 10 containing a series of rssi values (having already been averaged)
	:param function: method of coordinate calculation (knn/shepard)
	:param p: value for p
	:param k: value for k
	:return: 2*10 vector containing coordinate pairs for each rssi value
	"""
	coordinate_trajectory = [[0,0] for i in range(len(trajectory))]

	if function == sp.shepard_interpolation:
		for i, rssi_vector in enumerate(trajectory):
			coordinate_trajectory[i] = function(fingerprint, rssi_vector, p)

	elif function == knn.knn_regressor:
		for i, rssi_vector in enumerate(trajectory):
			coordinate_trajectory[i] = function(fingerprint, rssi_vector, k)

	return coordinate_trajectory
