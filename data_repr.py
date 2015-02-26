import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import data_fetch as df

# name of reader directory - used for filesystem path creation
r_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

# position of each reader - used for graph titles
r_pos = [' (-1,-1) ', ' (-1,7) ', ' (7,-1) ', ' (7,7) ']

def isolate_reader_data(data, reader):
	# this function takes in a 3D matrix of rssi data which has ALREADY
	# BEEN AVERAGED and returns a 2D matrix corresponding to the rssi data
	# from the reader specified by the 'reader' parameter

	reader_data = [[0 for i in range(len(data))] for i in range(len(data[0]))]

	for i in range(len(data)):
		for j in range(len(data[i])):
			reader_data[i][j] = data[i][j][reader]

	return reader_data

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
				plt.title('RSSI Over Time from Reader at{0}and Tag at ({1},{2})'.format(r_pos[r], str(i), str(j)))
				plt.savefig(dataset + r_dir[r] + str(i) + ',' + str(j) + '.svg', format='svg')
				plt.clf()

def plot_contour(dataset, rssi):
	rssi = df.rssi_average(rssi)
	
	for r in range(len(rssi[0][0])):
		reader_data = isolate_reader_data(rssi, r)

		fig = plt.subplot()
		image = plt.imshow(reader_data[:][::-1], interpolation='gaussian', cmap=cm.gray)
		colour_bar = plt.colorbar(image, orientation='horizontal')

		plt.title('Contour Map of Reader Positioned at ' + r_pos[r] + '\n')
		fig.set_xlabel('x Position')
		fig.xaxis.tick_top()
		plt.ylabel('y Position')
		plt.savefig(dataset + r_dir[r] + 'contour.svg', format='svg')
		plt.clf()