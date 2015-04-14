import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import data_fetch as df

# name of reader directory - used for filesystem path creation
r_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

# position of each reader - used for graph titles
r_pos = [' (-1,-1) ', ' (-1,7) ', ' (7,-1) ', ' (7,7) ']


def errors_histogram_show(errors):
	"""This function takes in a 2D array of error values which correspond to
	   the difference between the actual position and the predicted position
	   of the rssi reading by the location algorithm at each position on the
	   floor grid. It then unravels it, and creates a histogram of them."""
	matrix = np.asarray(errors)
	errors_list = np.reshape(matrix, -1)
	xbins = range(20)

	plt.hist(errors_list)
	plt.xlabel('Error margin (cm)')
	plt.title('Prediction Errors Frequency Distribution')
	plt.show()


def diagonal_rssi(dataset, rssi):
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


def errors_histogram_save(dataset, errors, k):
	matrix = np.asarray(errors)
	errors_list = np.reshape(matrix, -1)
	xbins = range(20)

	plt.hist(errors_list, bins=xbins)
	plt.xlabel('Error margin (cm)')
	plt.title('Prediction Errors Frequency Distribution')
	plt.savefig("{}/{}_errors_histogram.svg".format(dataset, str(k)), format='svg')
	plt.clf()


def point_rssi_wrt_time(dataset, rssi, t):
	for i in range(len(rssi)):
		for j in range(len(rssi[i])):
			for r in range(len(rssi[i][j])):
				# create line of best fit coefficients, then create line itself
				bf_c = np.polyfit(t[i][j][r], rssi[i][j][r], 1)
				bf = [x * bf_c[0] + bf_c[1] for x in t[i][j][r]]

				# plot raw data points
				plt.plot(t[i][j][r], rssi[i][j][r], 'bo-')

				# plot regression line
				plt.plot(t[i][j][r], bf, 'r--')

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


def trajectory_mse(predicted_path, path_type):
	actual_path = []
	if path_type.lower() == "parallel":
		actual_path = [[6-i*(0.7), 6-i*(0.7)] for i in range(10)]
	elif path_type.lower() == "perpendicular":
		actual_path = [[i*0.7, 6-i*0.7] for i in range(10)]

	mse = [(np.linalg.norm(np.array(pp) - np.array(ap)) * 50) for pp, ap in zip(predicted_path, actual_path)]

	bf_c = np.polyfit(range(len(predicted_path)), mse, 1)
	bf = [z * bf_c[0] + bf_c[1] for z in range(len(predicted_path))]

	# plot raw data points
	plt.plot(range(10), mse, 'bo-')

	# plot regression line
	plt.plot(range(10), bf, 'r--')

	plt.xlabel('Data Point')
	plt.ylabel('MSE (cm)')
	plt.title('Mean Squared Error At Each Point Along Trajectory')

	plt.savefig('diagonals/Trajectory MSE.svg', format='svg')
	plt.show()
	plt.clf()


def trajectory_overlay(predicted_path, path_type):
	actual_path = []
	if path_type.lower() == "parallel":
		actual_path = [[6 - i * (0.7), 6 - i * (0.7)] for i in range(10)]
	elif path_type.lower() == "perpendicular":
		actual_path = [[i * 0.7, 6 - i * 0.7] for i in range(10)]

	# line plotted depends on order of values in x list rather than magnitude -
	# this is good for plotting a trajectory
	plt.plot([p[0] for p in predicted_path], [p[1] for p in predicted_path], 'bo-')
	plt.plot([p[0] for p in actual_path], [p[1] for p in actual_path], 'ro-')

	plt.show()