__author__ = 'bapanna'

import data_fetch as df
import knn
import shepard as sp

def weighted_prediction(path, interpolator=knn.knn_regressor()):
	"""
	:param path: the vector of length 10 which contains the RSSI values taken
	over the trajectory
	:param interpolator: interpolation method used to find position of each RSSI value
	:return: a 10*2 matrix which contains the coordinates for the trajectory
	"""
	pass

t = df.fetch_diagonals()
parallel, perpendicular = df.trajectory_average(t)