import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import tests

import data_fetch as df
import weighted_predictor as wp
import shepard as sp
import knn
import kalman as km

# name of reader directory - used for filesystem path creation
r_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

# position of each reader - used for graph titles
r_pos = [' (-1,-1) ', ' (-1,7) ', ' (7,-1) ', ' (7,7) ']


def errors_histogram_show(errors):
	"""
	:param errors: 7*7 matrix of mse values for each point in the grid generated
	by an interpolation algorithm
	:return: histogram of error magnitude distribution
	"""
	matrix = np.asarray(errors)
	errors_list = np.reshape(matrix, -1)
	xbins = range(20)

	plt.hist(errors_list)
	plt.xlabel('Error margin (cm)')
	plt.title('Prediction Errors Frequency Distribution')
	plt.show()
	plt.clf()


def diagonal_rssi(dataset, rssi):
	"""
	:param dataset: dataset which the rssi values belong to
	:param rssi: 7*7*4 matrix of rssi information
	:return: plot of rssi values at each coordinate diagonally away from reader
	"""
	diagonal_coordinate_names = []
	diagonal_coordinate_names.append(['0,0', '1,1', '2,2', '3,3', '4,4', '5,5', '6,6'])
	diagonal_coordinate_names.append(['6,0', '5,1', '4,2', '3,3', '2,4', '1,5', '0,6'])
	diagonal_coordinate_names.append(diagonal_coordinate_names[1][::-1])
	diagonal_coordinate_names.append(diagonal_coordinate_names[0][::-1])

	rssi = df.fingerprint_average(rssi)

	for r in range(len(diagonal_coordinate_names)):
		values = []
		length = len(rssi) - 1

		if r == 0:
			for i in range(len(rssi)):
				values.append(rssi[i][i][r])
		elif r == 1:
			for i in range(len(rssi)):
				values.append(rssi[length - i][i][r])

		elif r == 2:
			for i in range(len(rssi)):
				values.append(rssi[i][length - i][r])

		elif r == 3:
			for i in range(len(rssi)):
				values.append(rssi[length - i][length - i][r])

		# create line of best fit coefficients, then create line itself
		bf_c = np.polyfit(range(len(rssi)), values, 1)
		bf = [z * bf_c[0] + bf_c[1] for z in range(len(rssi))]

		# plot raw data points
		plt.plot(range(7), values, 'bo-')

		# plot regression line
		plt.plot(range(7), bf, 'r--')

		# draw lines at y=0,100 to force axis scaling from 0 to 100
		plt.axhline(y=0)
		plt.axhline(y=100)

		plt.xlabel('Diagonal Coordinates from Reader Location')
		plt.ylabel('RSSI')
		plt.title('RSSI Against Diagonal Coordinates from Reader at ' + r_pos[r])

		plt.savefig(dataset + r_dir[r] + 'diagonal.svg', format='svg')
		plt.clf()


def errors_histogram_save(dataset, errors, k='', p=''):
	"""
	:param dataset: dataset to which the rssi values belong to
	:param errors: 7*7 matrix of mse values (real coordinate to predicted)
	:param k: k value for knn (if this method is used)
	:param p: p value for idw (if this method is used)
	:return: histogram showing distribution of mse magnitudes
	"""

	matrix = np.asarray(errors)
	errors_list = np.reshape(matrix, -1)
	xbins = range(20)

	plt.hist(errors_list)
	plt.xlabel('Error margin (cm)')

	if k:
		plt.title('Prediction Errors Frequency Distribution, k={}'.format(str(k)))
		plt.savefig("{}/{}_errors_histogram.svg".format(dataset, str(k)), format='svg')
	elif p:
		plt.title('Prediction Errors Frequency Distribution, p={}'.format(str(p)))
		plt.savefig("{}/{}_errors_histogram.svg".format(dataset, str(p)), format='svg')

	plt.clf()


def point_rssi_wrt_time(dataset, rssi, time):
	"""
	:param dataset: dataset to which the rssi values belong to
	:param rssi: 7*7*4*10 matrix of rssi values over 30s at each point for each
	reader
	:param time: 7*7*4*10 matrix of unix time values for each rssi value
	:return: plots of rssi value vs unix time for each point in the grid for
	each reader
	"""

	for i in range(len(rssi)):
		for j in range(len(rssi[i])):
			for r in range(len(rssi[i][j])):
				# create line of best fit coefficients, then create line itself
				bf_c = np.polyfit(time[i][j][r], rssi[i][j][r], 1)
				bf = [x * bf_c[0] + bf_c[1] for x in time[i][j][r]]

				# plot raw data points
				plt.plot(time[i][j][r], rssi[i][j][r], 'bo-')

				# plot regression line
				plt.plot(time[i][j][r], bf, 'r--')

				# draw lines at y=0,100 to force axis scaling from 0 to 100
				plt.axhline(y=0)
				plt.axhline(y=100)

				plt.xlabel('Unix Time')
				plt.ylabel('RSSI')
				plt.title(
					'RSSI Over Time from Reader at{0}and Tag at ({1},{2})'.format(
						r_pos[r], str(i), str(j)))
				plt.savefig(dataset + r_dir[r] + str(i) + ',' + str(j) + '.svg',
							format='svg')
				plt.clf()


def contour_plot(dataset, rssi):
	"""
	:param dataset: dataset to which the rssi values belong to
	:param rssi: 7*7*4*10 matrix of rssi values
	:return: heat map of rssi over the grid area for each reader
	"""
	rssi = df.fingerprint_average(rssi)

	for r in range(len(rssi[0][0])):
		reader_data = df.isolate_reader_data(rssi, r)

		fig = plt.subplot()
		image = plt.imshow(reader_data[:][::-1], interpolation='gaussian',
						   cmap=cm.gray)
		colour_bar = plt.colorbar(image, orientation='horizontal')

		plt.title('Contour Map of Reader Positioned at ' + r_pos[r] + '\n')
		fig.set_xlabel('x Position')
		fig.xaxis.tick_top()
		plt.ylabel('y Position')
		plt.savefig(dataset + r_dir[r] + 'contour.svg', format='svg')
		plt.clf()


def plot_trajectory_mse(predicted_path, path_type, method):
	"""
	:param predicted_path: trajectory which has been predicted using a Kalman
	filter or bespoke weighted predictor function
	:param path_type: string representing actual path which reader took
	:param method: interpolation method string for plot title
	:return: plot or errors at each point for trajectory
	"""
	actual_path = []
	if path_type.lower() == "parallel":
		actual_path = [[6-i*(0.6), 6-i*(0.6)] for i in range(10)]
	elif path_type.lower() == "perpendicular":
		actual_path = [[i*0.6, 6-i*0.6] for i in range(10)]

	mse = [(np.linalg.norm(np.array(pp) - np.array(ap)) * 50) for pp, ap in zip(predicted_path, actual_path)]
	log_mse = [0]
	for error in mse[1:]:
		log_mse.append(error)

	bf_c = np.polyfit(range(len(predicted_path)), log_mse, 1)
	bf = [z * bf_c[0] + bf_c[1] for z in range(len(predicted_path))]

	# plot raw data points
	plt.plot(range(1,11), log_mse, 'bo-')

	# plot regression line
	plt.plot(range(1,11), bf, 'r--')

	plt.xlabel('Data Point')
	plt.ylabel('MSE (cm)')
	plt.title('Mean Squared Error At Each Point Along Trajectory ({})\n'.format(method))

	plt.savefig('diagonals/{}_{}_Trajectory MSE.svg'.format(path_type, method[:4]), format='svg')
	plt.clf()


def plot_alpha_mse(path_rssi, fingerprint, path_type, function):
	"""
	:param path_rssi: list (length 10) of rssi values from trajectory measured
	:param fingerprint: 7*7*4 list of reference rssi values
	:param path_type: string representing path taken
	:param function: interpolation function
	:return: graph of mean mse for path vs different values for weighted
	predictor weight
	"""
	mean_mse = []

	for i in range(1,10):
		path_coords = wp.weighted_prediction(path_rssi, fingerprint, path_type, function=function, w=i/10)
		path_error = tests.trajectory_mse(path_coords, w=i/10)
		mean_mse.append(np.array(path_error).mean())

	x = np.array(range(1, 10)) / 10

	plt.plot(x, mean_mse, 'bo-')
	plt.xlabel('Alpha')
	plt.ylabel('MSE (cm)')
	plt.axhline(y=0.1)

	method = ''
	scheme = ''

	if function == sp.shepard_interpolation:
		method = 'IDW, p=3'
		scheme = 'shepard'

	elif function == knn.knn_regressor:
		method = 'KNN, k=11'
		scheme = 'knn'

	plt.title('{} Path Mean MSE per Alpha vs. Alpha ({})'.format(path_type.title(), method))
	plt.savefig('diagonals/{} {} mse across alpha.svg'.format(scheme, path_type), format='svg')
	plt.clf()


def trajectory_overlay(predicted_path, path_type, suffix):
	"""
	:param predicted_path: list of length 10 containing coordinate pairs of predicted trajectory
	:param path_type: string representing actual path taken by reader
	:param suffix: additional string used for graph title
	:return: plot of actual and predicted trajectory
	"""
	actual_path = []
	if path_type.lower() == "parallel":
		actual_path = [[6 - i * (0.7), 6 - i * (0.7)] for i in range(10)]
	elif path_type.lower() == "perpendicular":
		actual_path = [[i * 0.7, 6 - i * 0.7] for i in range(10)]

	# line plotted depends on order of values in x list rather than magnitude -
	# this is good for plotting a trajectory
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	ax.plot([p[0] for p in predicted_path], [p[1] for p in predicted_path], color='r', label='Predicted Path')
	ax.plot([p[0] for p in actual_path], [p[1] for p in actual_path], color='b', label='Actual Path')

	ax.plot([p[0] for p in predicted_path], [p[1] for p in predicted_path], color='r', linestyle='', marker='o')
	ax.plot([p[0] for p in actual_path], [p[1] for p in actual_path], color='b', linestyle='', marker='o')

	for i, xy in enumerate(predicted_path):
		ax.annotate(str(i+1), xy=xy, textcoords = 'offset points')

	for i, xy in enumerate(actual_path):
		ax.annotate(str(i + 1), xy=xy, textcoords='offset points')

	plt.gca().invert_yaxis()

	handles, labels = ax.get_legend_handles_labels()
	display = (0,1)

	simArtist = plt.Line2D((0, 1), (0, 0), color='k', marker='o', linestyle='')
	anyArtist = plt.Line2D((0, 1), (0, 0), color='k')

	ax.legend([handle for i, handle in enumerate(handles) if i in display] + [simArtist, anyArtist], [label for i, label in enumerate(labels) if i in display])
	ax.xaxis.tick_top()

	plt.title('Reader Trajectories Using {}, {} Path{}'.format(suffix, path_type.title(), '\n\n'))
	plt.grid()

	if suffix[0] == 'I':
		plt.savefig('diagonals/trajectory weighted {} shepard.svg'.format(path_type), format='svg')

	else:
		plt.savefig('diagonals/trajectory weighted {} knn.svg'.format(path_type), format='svg')

	# plt.show()
	plt.clf()

# using only elevated dataset because trajectory data was taken under similar conditions

# retrieving raw data
strengths, times = df.fetch_fingerprint("elevated")
fingerprint = df.fingerprint_average(strengths)
diagonals = df.fetch_diagonals()
parallel, perpendicular = df.trajectory_average(diagonals)

# rssi values converted to cartesian coordinates
parallel_path_sp = km.rssi_to_coordinates(fingerprint, parallel, sp.shepard_interpolation)
parallel_path_knn = km.rssi_to_coordinates(fingerprint, parallel, knn.knn_regressor)
perpendicular_path_sp = km.rssi_to_coordinates(fingerprint, parallel, sp.shepard_interpolation)
perpendicular_path_knn = km.rssi_to_coordinates(fingerprint, parallel, sp.shepard_interpolation)

# filtered coordinate pairs
k_parallel_sp = km.multivariate_kalman(parallel_path_sp)
k_parallel_knn = km.multivariate_kalman(parallel_path_knn)
k_perpendicular_sp = km.multivariate_kalman(perpendicular_path_sp)
k_perpendicular_knn = km.multivariate_kalman(perpendicular_path_knn)

# FIX MULTIVARIATE KALMAN OUTPUT DATA FORMAT

# plot filtered coordinate pairs
# plot_trajectory_mse(k_parallel_sp, 'parallel', 'IDW, p=3')
# plot_trajectory_mse(k_parallel_knn, 'parallel', 'KNN, k=11')
# plot_trajectory_mse(k_perpendicular_sp, 'perpendicular', 'IDW, p=3')
# plot_trajectory_mse(k_perpendicular_knn, 'perpendicular', 'KNN, k=11')

# plot trajectories
trajectory_overlay(k_parallel_sp, 'parallel', 'IDW, p=3')
trajectory_overlay(k_parallel_knn, 'parallel', 'KNN, k=11')
trajectory_overlay(k_perpendicular_sp, 'perpendicular', 'IDW, p=3')
trajectory_overlay(k_perpendicular_knn, 'perpendicular', 'KNN, k=11')