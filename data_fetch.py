__author__ = 'bapanna'

import numpy as np
import math


# OK
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


# OK
def rssi_average(strengths):
	# input will be array of individual values
	for i in range(len(strengths)): # 7
		for j in range(len(strengths[i])): # 7
			for r in range(len(strengths[i][j])): # 4
				strengths[i][j][r] = np.median(strengths[i][j][r])

	return strengths


def convert_trajectories_form(traj):
	new_traj = [[[[] for i in range(4)] for i in range(10)] for i in range(2)]

	for d in range(len(new_traj)):
		for p in range(len(new_traj[d])):
			for r in range(len(new_traj[d][p])):
				for t in range(len(traj[d][r])):
					new_traj[d][p][r].append(traj[d][r][t][p])

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


# OK
def fetch_fingerprint(dataset):
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