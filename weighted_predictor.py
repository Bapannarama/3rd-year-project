__author__ = 'bapanna'

import data_fetch as df
import data_repr as dr
import knn
import numpy as np
import shepard as sp

def weighted_prediction(path, fingerprint, direction, w=0.5):
	"""
	:param path: a 10*4 matrix which contains the RSSI values taken
	over the trajectory
	:param weight: weight of knn prediction for prediction function
	:return: a 10*2 matrix which contains the coordinates for the trajectory

	The knn operation is naïve and does not take into account the previous
	position
	"""
	trajectory = []

	if direction.lower() == "parallel":
		trajectory.append([6,6])

	elif direction.lower() == "perpendicular":
		trajectory.append([0,6])

	# this line adds the second point to the trajectory matrix - allows the
	# predictor which assumes constant movement to function
	trajectory.append(knn.nn_classifier(fingerprint, path[1]))

	for i in range(1,10):
		nn_prediction = knn.nn_classifier(fingerprint, path[i])
		# predicts next position based on previous step's movement
		const_movement_prediction = [(2 * p_1 - p_0) for p_1, p_0 in zip(trajectory[i], trajectory[i - 1])]
		final_prediction = [(w*k + (1-w)*c) for k,c in zip(nn_prediction, const_movement_prediction)]
		trajectory.append(final_prediction)

	return trajectory

strengths, times = df.fetch_fingerprint('elevated')
fingerprint = df.fingerprint_kalman(strengths)
t = df.fetch_diagonals()
parallel, perpendicular = df.trajectory_average(t)
parallel_path = weighted_prediction(parallel, fingerprint, "parallel", 3)
print(parallel_path)
# dr.trajectory_mse(parallel_path, "parallel")