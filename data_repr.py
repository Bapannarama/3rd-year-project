import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import data_fetch as df

# name of reader directory - used for filesystem path creation
r_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

# position of each reader - used for graph titles
r_pos = [' (-1,-1) ', ' (-1,7) ', ' (7,-1) ', ' (7,7) ']


def diagonal_rssi(dataset, rssi):
	diagonal_coordinate_names = []
	diagonal_coordinate_names.append(
		['0,0', '1,1', '2,2', '3,3', '4,4', '5,5', '6,6'])
	diagonal_coordinate_names.append(
		['6,0', '5,1', '4,2', '3,3', '2,4', '1,5', '0,6'])
	diagonal_coordinate_names.append(diagonal_coordinate_names[1][::-1])
	diagonal_coordinate_names.append(diagonal_coordinate_names[0][::-1])

	rssi = df.rssi_average(rssi)

	for r in range(len(diagonal_coordinate_names)):
		values = []
		reader_values = df.isolate_reader_data(rssi, r)

		if r == 0:
			for i in range(len(rssi)): # loops 7 times dw mate
				values.append(reader_values[i][i])

		elif r == 1:
			for i in range(len(rssi)):
				values.append(reader_values[len(reader_values) - i][i])

		elif r == 2:
			for i in range(len(rssi)):
				values.append(rssi[i][len(rssi) - i])

		elif r == 3:
			for i in range(len(rssi)):
				values.append(rssi[len(rssi) - i][len(rssi) - i])

		# create line of best fit coefficients, then create line itself
		bf_c = np.polyfit(range(len(reader_values)), values, 1)
		bf = [z * bf_c[0] + bf_c[1] for z in range(len(reader_values))]

		# plot raw data points
		plt.plot(range(7), values, 'bo-')

		# plot regression line
		plt.plot(range(7), bf, 'r--')

		# draw lines at y=0,100 to force axis scaling from 0 to 100
		plt.axhline(y=0)
		plt.axhline(y=100)

		plt.xlabel('Diagonal Coordinates from Reader Location')
		plt.ylabel('RSSI')
		plt.title(
			'RSSI Against Diagonal Coordinates from Reader at ' + r_pos[r])

		plt.savefig(dataset + r_dir[r] + 'diagonal.svg', format='svg')
		plt.clf()


def errors_histogram(errors):
	matrix = np.asarray(errors)
	errors_list = np.reshape(matrix, -1)
	xbins = range(20)

	plt.hist(errors_list, bins=xbins)
	plt.xlabel('Error margin (cm)')
	plt.show()


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


def contour(dataset, rssi):
	rssi = df.rssi_average(rssi)

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