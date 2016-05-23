import numpy as np
import pandas as pd
import os

def load_dataset():
	filename = os.path.join(os.path.dirname(__file__), "adult.data.txt")

	# dtypes = {
	# 	"names": ("age", "workplace")
	# 	"formats": (int, str, int, str, int, str, str, str, str, str, int, int, int, str)}

	data = np.genfromtxt(filename, delimiter=",", dtype=None)
	data2 = pd.DataFrame(data).as_matrix()

	# print data2.dtypes
	# print data2.shape

	X, Y = data2[:,:-1], data2[:,-1]

	# data2 = map(lambda x: (list(x)[:-1], list(x)[-1]), data)

	# X = map(lambda x: x[0], data2)
	# Y = map(lambda x: x[1], data2)

	return X, Y