__author__ = 'bapanna'

import data_fetch as df
import data_repr as dr
import numpy as np
import shepard as sp
import knn
import kalman as km
import matplotlib.pyplot as plt


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


dataset = "reader_ground_level"
strengths, times = df.fetch_fingerprint(dataset)
fingerprint = df.fingerprint_average(strengths)

mse = []

for i in range(1,50):
	errors = generate_errors_matrix(fingerprint, sp.shepard_interpolation, p=i)
	print(np.array(errors).mean())
	mse.append(np.array(errors).mean())
	dr.errors_histogram_save(dataset, errors, i)

plt.plot(range(1,50), mse, 'bo-')
plt.axhline(y=120)
plt.xlabel("k")
plt.ylabel("Mean Squared Error (cm²)")
plt.title("Mean Squared Error vs. k (KNN)")
plt.savefig("{}/knn k errors.svg".format(dataset), format='svg')

# elevated best knn error: k=11 @ 130.069211329
# elevated best shepard error: p=3 @ 122.511396273

# reader_ground best shepard error: p=1 @ 136.177528736
# reader_ground best knn error: k=49 @ 132.624796737