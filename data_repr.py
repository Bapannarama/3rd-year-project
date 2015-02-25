import matplotlib.pyplot as plt
import numpy as np

# name of reader directory - used for filesystem path creation
r_dir = ['/0,0/', '/0,6/', '/6,0/', '/6,6/']

# position of each reader - used for graph titles
r_pos = [' (-1,-1) ', ' (-1,7) ', ' (7,-1) ', ' (7,7) ']

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