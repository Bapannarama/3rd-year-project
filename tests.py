__author__ = 'bapanna'

import data_fetch as df
import data_repr as dr
import numpy as np

dataset = 'elevated'
s, t = df.fetch_data(dataset)
dr.diagonal_rssi(dataset, s)