__author__ = 'bapanna'

# TODO: implement threading into load_data

from matplotlib import cm
import matplotlib.pyplot as pp
import numpy as np

# helper function which returns an averaged matrix of rssis
def setup_elevated():
	reader_directory_names = ['0,0/', '0,6/', '6,0/', '6,6/'] # used for filenames
	strengths, times = load_data('elevated', 10, reader_directory_names)
	strengths = coordinate_average(strengths)
	return strengths

def generate_errors_histogram(matrix):
	matrix = np.asarray(matrix)
	errors_list = np.reshape(matrix, -1)
	xbins = range(20)

	pp.hist(errors_list)
	pp.xlabel('Error margin (cm)')
	pp.show()

def plot_diagonal(trial_config, RSSI, r_dir_names, r_pos):
	diagonal_coordinate_names = []
	diagonal_coordinate_names.append(['0,0', '1,1', '2,2', '3,3', '4,4', '5,5', '6,6'])
	diagonal_coordinate_names.append(['6,0', '5,1', '4,2', '3,3', '2,4', '1,5', '0,6'])
	diagonal_coordinate_names.append(diagonal_coordinate_names[1][::-1])
	diagonal_coordinate_names.append(diagonal_coordinate_names[0][::-1])

	RSSI = coordinate_average(RSSI)

	for r in range(len(RSSI)):
		vals = []

		if r == 0:
			for i in range(len(RSSI[0])):
				vals.append(RSSI[r][i][i])

		elif r == 1:
			for i in range(len(RSSI[0])):
				vals.append(RSSI[r][len(RSSI) - i][i])

		elif r == 2:
			for i in range(len(RSSI[0])):
				vals.append(RSSI[r][i][len(RSSI) - i])

		elif r == 3:
			for i in range(len(RSSI[0])):
				vals.append(RSSI[r][len(RSSI) - i][len(RSSI) - i])

		# polyfit creates a least squares regression line
		regressor = np.polyfit(range(7), vals, 1)
		#bestfit = line of best fit
		best_fit = [z * regressor[0] + regressor[1] for z in range(7)]

		# plot raw data points
		pp.plot(range(7), vals, 'bo-')
		# plot regression line
		pp.plot(range(7), best_fit, 'r--')
		# draw lines at y=0,100 to force axis scaling from 0 to 100
		pp.axhline(y=0)
		pp.axhline(y=100)
		pp.xlabel('Diagonal Coordinates from Reader Location')
		pp.ylabel('RSSI')
		pp.title('RSSI Against Diagonal Coordinates from Reader at ' + r_pos[r])

		pp.savefig(trial_config + '/' + r_dir_names[r] + 'diagonal.svg', format='svg')
		pp.clf()


def plot_contour_old(trial_config, RSSI, r_dir_names, r_pos):
	RSSI = coordinate_average(RSSI)
	# data at strength[x][y][z] is now a float rather than a list

	for r in range(4):
		image = pp.imshow(RSSI[r][:][::-1], interpolation='bilinear',
						  cmap=cm.gray)
		colour_bar = pp.colorbar(image, orientation='horizontal')

		# pp.axis([0, 6, 6, 0]) # hack to make axes orientate themselves correctly
		pp.xlabel('x Position')
		image.xaxis.set_label_position('top')
		pp.ylabel('y Position')
		pp.title('Contour Map of Reader Positioned at ' + r_pos[r] + '\n\n\n')
		pp.savefig(trial_config + '/' + r_dir_names[r] + "contour.svg",
				   format='svg')
		pp.clf()


def plot_contour(trial_config, RSSI, r_dir_names, r_pos):
	RSSI = coordinate_average(RSSI)

	for r in range(4):
		fig = pp.subplot()
		image = pp.imshow(RSSI[r][:][::-1], interpolation='gaussian',
						  cmap=cm.gray)
		colour_bar = pp.colorbar(image, orientation='horizontal')

		# pp.axis([0, 6, 6, 0]) # hack to make axes orientate themselves correctly
		pp.title('Contour Map of Reader Positioned at ' + r_pos[r] + '\n')
		fig.set_xlabel('x Position')
		# fig.xaxis.set_label_position('top')
		fig.xaxis.tick_top()
		pp.ylabel('y Position')
		pp.savefig(trial_config + '/' + r_dir_names[r] + "contour.svg",
				   format='svg')
		pp.clf()

# returns list of median
def coordinate_average(data):
	for r in range(4):
		for x in range(7):
			for y in range(7):
				# calculate the median signal strength at each point for each
				# reader
				data[r][x][y] = np.median(data[r][x][y])

	return data


def plot_data(trial_config, RSSI, t, r_dir_names, r_pos):
	for r in range(4):
		for x in range(7):
			for y in range(7):
				# polyfit creates a least squares regression line
				regressor = np.polyfit(t[r][x][y], RSSI[r][x][y], 1)
				# best_fit = line of best fit
				best_fit = [z * regressor[0] + regressor[1] for z in t[r][x][y]]

				# plot raw data points
				pp.plot(t[r][x][y], RSSI[r][x][y], 'bo-')
				# plot regression line
				pp.plot(t[r][x][y], best_fit, 'r--')

				# draw lines at y=0,100 to force axis scaling from 0 to 100
				pp.axhline(y=0)
				pp.axhline(y=100)
				pp.xlabel('Unix Time')
				pp.ylabel('RSSI')
				pp.title('RSSI Against Time of Reader at ' + \
					r_pos[r] + ' and Tag at ' + '(' + str(x) + ',' +
						 str(y) + ')')
				pp.savefig(trial_config + '/' + r_dir_names[r] + str(x) + ',' +
						   str(y) + '.svg', format='svg')
				pp.clf()


# function goes through each test file taking out RSSI and time values from raw
# results and storing them in a list. These are then appended to a list containing
# all y values, which is then appended to one containing all x values, which is
# then appended to one for each reader.
# The final format for the resulting matrices is [reader][x][y]
def load_data(trial_config, measurements, r_dir_names):
	RSSI = []
	t = []

	for r in range(4):
		tmp_x_RSSI = []
		tmp_x_t = []

		# collects all x values
		for x in range(7):
			tmp_y_RSSI = []
			tmp_y_t = []

			# collects all y values
			for y in range(7):
				filename = trial_config + '/' + r_dir_names[r] + str(x) + ',' +\
						   str(y) + '.txt'
				file = open(filename, 'r')
				tmp_coordinate_RSSI = []
				tmp_coordinate_t = []

				# collects data for each point by appending each line
				for line_no in range(measurements):
					line = file.readline()
					tmp_coordinate_RSSI.append(int(line[4:6]))
					tmp_coordinate_t.append(int(line[7:21]))

				tmp_y_RSSI.append(tmp_coordinate_RSSI)
				tmp_y_t.append(tmp_coordinate_t)

			tmp_x_RSSI.append(tmp_y_RSSI)
			tmp_x_t.append(tmp_y_t)

		RSSI.append(tmp_x_RSSI)
		t.append(tmp_x_t)

	return RSSI, t


reader_directory_names = ['0,0/', '0,6/', '6,0/', '6,6/'] # used for filenames
reader_positions = ['(0,0)', '(0,6)', '(6,0)', '(6,6)'] # used for graph titles

dataset = 'reader_ground_level'
strengths, times = load_data(dataset, 10, reader_directory_names)
plot_data(dataset, strengths, times, reader_directory_names, reader_positions)
plot_contour(dataset, strengths, reader_directory_names, reader_positions)
plot_diagonal(dataset, strengths, reader_directory_names, reader_positions)

dataset = 'reader,tag_ground_level'
strengths, times = load_data(dataset, 5, reader_directory_names)
plot_data(dataset, strengths, times, reader_directory_names, reader_positions)
plot_contour(dataset, strengths, reader_directory_names, reader_positions)
plot_diagonal(dataset, strengths, reader_directory_names, reader_positions)

# used for graph titles
reader_positions = ['(-1,-1)', '(-1,7)', '(7,-1)', '(7,7)']
dataset = 'elevated'
strengths, times = load_data(dataset, 10, reader_directory_names)
plot_data(dataset, strengths, times, reader_directory_names, reader_positions)
plot_contour(dataset, strengths, reader_directory_names, reader_positions)
plot_diagonal(dataset, strengths, reader_directory_names, reader_positions)

# dataset = dataset
# reader_folders = reader_directory_names / r_dir_names
# reader_positions = reader_positions / r_pos
# strengths = strengths / RSSI
# times = times / t