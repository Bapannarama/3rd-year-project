__author__ = 'bapanna'

import numpy as np
import kalman as km
import math, statistics


def dummy_data():
	# empty array filled with zeroes which will be populated properly later
	strengths = [[[0 for i in range(4)] for i in range(7)] for i in range(7)]

	# range arguments must now go from 1 to 7 as readers are outside test area
	for i in range(1,8):
		for j in range(1,8):
			strengths[i-1][j-1][0] = math.sqrt(i**2 + j**2)
			strengths[i-1][j-1][1] = math.sqrt((8-i)**2 + j**2)
			strengths[i-1][j-1][2] = math.sqrt(i**2 + (8-j)**2)
			strengths[i-1][j-1][3] = math.sqrt((8-i)**2 + (8-j)**2)

	return strengths


def fingerprint_average(strengths):
	# input will be array of individual values
	for i in range(len(strengths)): # 7
		for j in range(len(strengths[i])): # 7
			for r in range(len(strengths[i][j])): # 4
				strengths[i][j][r] = np.median(strengths[i][j][r])

	return strengths


def fingerprint_kalman(strengths):
	# input will be array of individual values
	for i in range(len(strengths)): # 7
		for j in range(len(strengths[i])): # 7
			for r in range(len(strengths[i][j])): # 4
				strengths[i][j][r] = km.univariate_kalman(strengths[i][j][r])

	return strengths


def trajectory_average(t):
	"""
	finds the meadian for each data point of each reader across the three test values
	"""

	for p in range(len(t)):
		for pt in range(len(t[p])):
			for r in range(len(t[p][pt])):
				t[p][pt][r] = statistics.median(t[p][pt][r])

	return t


def convert_trajectories_form(traj):
	new_traj = [[[[0,0,0] for i in range(4)] for i in range(10)] for i in range(2)]

	# we want to collect all the test results for a particular point from a reader
	# collecting info from traj

	for paths in range(len(traj)): # 2
		for readers in range(len(traj[paths])): # 4
			for test in range(len(traj[paths][readers])): # 3
				for point in range(len(traj[paths][readers][test])): # 10
					new_traj[paths][point][readers][test] = traj[paths][readers][test][point]

	return new_traj


def fetch_diagonals():
	"""
	Function fetches information on diagonals trajectory tests and stores in a
	list
	"""

	# path to location where data for trajectory tests are contained
	traj_dir = ["diagonals/parallel", "diagonals/perpendicular"]
	# path inside traj_dir where data from each of the readers is contained
	reader_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

	trajectories = [[[[] for i in range(3)] for i in range(4)] for i in range(2)]

	for p in range(len(trajectories)):
		for r in range(len(trajectories[p])):
			for t in range(len(trajectories[p][r])):
				filename = str(t+1) + ".txt"
				path = traj_dir[p] + reader_dir[r] + filename
				file = open(path, 'r')
				line = file.readline()

				while not(line == ""):
					trajectories[p][r][t].append(int(line[4:6]))
					line = file.readline()

	# trajectories is current in format [path][reader][test][point]
	# we want it in form [path][point][reader][test]
	trajectories = convert_trajectories_form(trajectories)

	return trajectories


def fetch_fingerprint(dataset):
	"""
	:param dataset: name of the folder which the fingerprint data is stored in
	:return: a matrix of the form fingerprint[x][y][reader][rssi_across_time]
	"""
	# dataset will be name of directory containing data
	reader_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

	strengths = [[[[] for i in range(4)] for i in range(7)] for i in range(7)]
	times     = [[[[] for i in range(4)] for i in range(7)] for i in range(7)]

	for i in range(len(strengths)):
		for j in range(len(strengths[i])):
			for r in range(len(strengths[i][j])):
				filename = str(i) + ',' + str(j) + '.txt'
				path = dataset + reader_dir[r] + filename
				file = open(path, 'r')
				line = file.readline()

				while not(line == ''):
					strengths[i][j][r].append(int(line[4: 6]))
					times    [i][j][r].append(int(line[7:-1]))
					line = file.readline()

	return strengths, times


def isolate_reader_data(data, reader):
	# this function takes in a 3D matrix of rssi data which has ALREADY
	# BEEN AVERAGED and returns a 2D matrix corresponding to the rssi data
	# from the reader specified by the 'reader' parameter

	reader_data = [[0 for i in range(len(data))] for i in range(len(data[0]))]

	for i in range(len(data)):
		for j in range(len(data[i])):
			reader_data[i][j] = data[i][j][reader]

	return reader_data