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

dataset = "wine"  # linearReg, wine

if dataset == "wine":
    data = pd.read_csv("datasets/winequality-red.csv", delimiter=";")
    truth = data["alcohol"].as_matrix()
    cols = ['fixed acidity', 'pH', 'density', 'chlorides']
    # cols = ['fixed acidity']
    features = data[cols].as_matrix()
    # Normalize histogramms
    features = normalize_features(features)

    # measures = truth + np.random.normal(0, 0.2, len(truth))
    train_features = features[:750, :]
    train_measures = truth[:750]
    # truth = truth[:750]

    test_features = features[750:1000, :]
    test_measures = truth[750:1000]

elif dataset == "linearReg":
    features = np.linspace(-np.pi, np.pi, 50)
    features = features[:, None]
    truth = np.sin(features)
    # truth = features*np.sin(features*0.1)
    measures = truth + np.random.normal(0, 0.2, [len(truth), 1])

    test_features = np.linspace(-np.pi, np.pi, 100)
    test_features = test_features[:, None]
    # test_points_y = [0.25, 26]


kernel = get_kernel(train_features, test_features, train_measures)

test_size = test_features.shape[0]
train_size = train_features.shape[0]

print kernel.shape

test_kernel = kernel[train_size:, :train_size]
train_kernel = kernel[:train_size, :train_size]

plt.imsave("kernel.png", kernel)

def my_kernel(X, Y):
    return kernel

# print(kernel.shape)
plt.imshow(kernel)


##################################################################
svr_rf = SVR(kernel="precomputed")
# svm_c = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
svr_rf.fit(train_kernel, train_measures)
print(svr_rf.intercept_)


##################################################################
C = np.linspace(0.1, 100, 6)
gamma = np.logspace(-2, 3, 6)
print('Performing cross-validation for SVR parameters...')
Copt, gammaOpt = select_parameters(C, gamma, train_features, train_measures)

print Copt, gammaOpt

svr_rbf = SVR(kernel='rbf', C=Copt, gamma=gammaOpt)
# svm_c = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
svr_rbf.fit(train_features, train_measures)

##################################################################
if dataset == 'linearReg':
    fig, ax = plt.subplots()
    ax.plot(train_features, truth, label='Ground truth')
    ax.scatter(train_features, train_measures, label='Points')

    data = np.concatenate((features, test_features), axis=0)

    result = svr_rf.predict(test_kernel)
    print result

    ax.plot(test_features, result, label='Random Forest kernel')
    # ax.plot(features, svr_rbf.predict(features), label='RBF')
    legend = ax.legend()
    plt.show()
    fig.savefig("plot.png")

elif dataset == 'wine':
    RF_predict = svr_rf.predict(test_kernel)
    RBF_predict = svr_rbf.predict(test_features)
    r2_rf = r2_score(test_measures, RF_predict)
    mse_rf = np.mean((test_measures - RF_predict) ** 2)
    r2_rbf = r2_score(test_measures, RBF_predict)
    mse_rbf = np.mean((test_measures - RBF_predict) ** 2)

    print(mse_rf)
    print(mse_rbf)

    fig, ax = plt.subplots()
    ax.scatter(test_measures, RF_predict)
    plt.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2_rf), c="r")
    plt.legend(loc="lower right")
    plt.title('RF kernel')
    plt.show
    fig, ax2 = plt.subplots()
    ax2.scatter(test_measures, RBF_predict)
    plt.plot(np.arange(8, 15), np.arange(8, 15), label="r^2=" + str(r2_rbf), c="r")
    plt.legend(loc="lower right")
    plt.title('RBF kernel')
    plt.show()
