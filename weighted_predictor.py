__author__ = 'bapanna'

import knn
import shepard as sp


def weighted_prediction(path, fingerprint, direction, function = knn.knn_regressor, w=0.5):
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
	trajectory.append(function(fingerprint, path[1]))

	for i in range(1,9):
		nn_prediction = function(fingerprint, path[i])
		# predicts next position based on previous step's movement
		const_movement_prediction = [(2 * p_1 - p_0) for p_1, p_0 in zip(trajectory[i], trajectory[i - 1])]
		final_prediction = [(w*k + (1-w)*c) for k,c in zip(nn_prediction, const_movement_prediction)]
		trajectory.append(final_prediction)

	return trajectory

"""
Best weight values:
parallel shepard = 0.5
parallel knn = 0.5
perpendicular shepard = 0.5
perpendicular knn = 0.5
"""