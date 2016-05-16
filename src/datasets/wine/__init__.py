import numpy as np
import os

def load_dataset():
	filename = os.path.join(os.path.dirname(__file__), "wine.data.txt")

	data = np.loadtxt(filename, delimiter=",")

	return data[:, 1:], data[:, 0]
