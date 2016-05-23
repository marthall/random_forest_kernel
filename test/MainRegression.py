# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:25:49 2016

@author: lzeng
"""
import pandas as pd
# import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
from RegressionKernel import get_kernel
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from func import *

dataset = "linearReg"  # linearReg, wine

if dataset == "wine":
    data = pd.read_csv("datasets/winequality-red.csv", delimiter=";")
    truth = data["alcohol"].as_matrix()
    cols = ['fixed acidity', 'pH', 'density', 'chlorides']
    # cols = ['fixed acidity']
    features = data[cols].as_matrix()
    measures = truth + np.random.normal(0, 0.2, len(truth))
    features = features[0:750, :]
    measures = measures[0:750]
    truth = truth[0:750]
    # Normalize histogramms
    features = normalize_features(features)
elif dataset == "linearReg":
    features = np.linspace(-10, 10, 100)
    features = features[:, None]
    truth = features ** 2
    # truth = features*np.sin(features*0.1)
    measures = truth + np.random.normal(0, 3, [len(truth), 1])

    test_points_x = np.array([[0.56,], [5.42,]])
    test_points_y = [0.25, 26]


kernel = get_kernel(features, test_points_x, measures)

plt.imsave("kernel.png", kernel)

def my_kernel(X, Y):
    return kernel

# print(kernel.shape)
plt.imshow(kernel)


##################################################################
svr_rf = SVR(kernel=my_kernel)
# svm_c = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
svr_rf.fit(features, measures)
print(svr_rf.intercept_)


##################################################################
C = np.linspace(0.1, 100, 6)
gamma = np.logspace(-2, 3, 6)
print('Performing cross-validation for SVR parameters...')
Copt, gammaOpt = select_parameters(C, gamma, features, measures)

print Copt, gammaOpt

svr_rbf = SVR(kernel='rbf', C=Copt, gamma=gammaOpt)
# svm_c = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
svr_rbf.fit(features, measures)

##################################################################
if dataset == 'linearReg':
    fig, ax = plt.subplots()
    ax.plot(features, truth, label='Ground truth')
    ax.scatter(features, measures, label='Points')

    data = np.concatenate((features, test_points_x), axis=0)

    ax.plot(features, svr_rf.predict(features), label='Random Forest kernel')
    ax.plot(features, svr_rbf.predict(features), label='RBF')
    legend = ax.legend()
    plt.show()
    fig.savefig("plot.png")

elif dataset == 'wine':
    RF_predict = y_rf.predict(features)
    RBF_predict = y_rbf.predict(features)
    r2_rf = r2_score(truth, RF_predict)
    mse_rf = np.mean((truth - RF_predict) ** 2)
    r2_rbf = r2_score(truth, RBF_predict)
    mse_rbf = np.mean((truth - RBF_predict) ** 2)

    fig, ax = plt.subplots()
    ax.scatter(truth, RF_predict)
    plt.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2_rf), c="r")
    plt.legend(loc="lower right")
    plt.title('RF kernel')
    plt.show
    fig, ax2 = plt.subplots()
    ax2.scatter(truth, RBF_predict)
    plt.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2_rbf), c="r")
    plt.legend(loc="lower right")
    plt.title('RBF kernel')
    plt.show()
