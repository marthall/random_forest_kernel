import numpy as np
import os

def load_dataset():
	filename = os.path.join(os.path.dirname(__file__), "isolet1+2+3+4.data")

	data = np.loadtxt(filename, delimiter=",")

	return data[:, :-1], data[:, -1]
