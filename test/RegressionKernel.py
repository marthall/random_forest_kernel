# -*- coding: utf-8 -*-
"""
Created on Thu May 19 12:21:24 2016

@author: lzeng
"""

import pandas as pd
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestRegressor
from itertools import compress
from func import get_leaves
from func import get_lineage

def get_partial_kernel(forest, dataset):
    SAMPLE_SIZE = len(dataset)

    #Select random tree
    TreeIndex = randint(0,forest.n_estimators-1)
    Arbre = forest.estimators_[TreeIndex]
    #Define tree parameters
    n_nodes = Arbre.tree_.node_count
    children_left = Arbre.tree_.children_left
    children_right = Arbre.tree_.children_right

    # h =Arbre.tree_.max_depth #all trees don't have the same height???

    #Node selection
    leaf_index = Arbre.apply(dataset)
    leaf_nodes = np.array(list(set(leaf_index)))  # Remove duplicates
    # print "Leaf index size: {}".format(len(leaf_nodes))

    is_leaves, node_depth = get_leaves(children_left, children_right, n_nodes)
    # print "is_leaves: {}".format(sum(is_leaves))
    myparents = get_lineage(Arbre, leaf_index, node_depth)

    myparents = np.array(myparents)

    h = max(node_depth[is_leaves]) - 1
    d = randint(0, h)

    # print(d)

    # print(np.mean(node_depth[is_leaves]))
    # print(np.var(node_depth[is_leaves]))
    # print "Max-height: {}".format(h)
    # print "Choses-height: {}".format(d)

    nth_parent_for_sample = []
    for lineage in myparents:
        ancest = None
        if len(lineage) <= d:
            ancest = lineage[-1]
        else:
            ancest = lineage[d]
        nth_parent_for_sample.append(ancest)

    partial_kernel = np.zeros([SAMPLE_SIZE, SAMPLE_SIZE])

    #Loop over each datapoint : two data point are assigned to the same cluster if they have the same ancestor
    for i in range(SAMPLE_SIZE):
        for j in range(i, SAMPLE_SIZE):
            if nth_parent_for_sample[i][0] == nth_parent_for_sample[j][0]:
                partial_kernel[i][j] = 1
                partial_kernel[j][i] = 1

    return partial_kernel


def get_kernel(train_data, test_data, label):

    #Define forest (n_estimators = number of trees)
    forest = RandomForestRegressor(n_estimators=1000, warm_start = True)
    forest = forest.fit(train_data, label)

    dataset = np.concatenate((train_data, test_data), axis=0)

    SAMPLE_SIZE = len(train_data)
    M = 400

    #Loop that generates samples of the PDF
    kernel_list = np.empty([M, SAMPLE_SIZE, SAMPLE_SIZE])
    for m in range(M):
        # print("Building partial kernel: {}".format(m))
        kernel_list[m,:,:] = get_partial_kernel(forest, train_data)

    #Average the samples to compute the kernel
    kernel = np.mean(kernel_list, axis=0)

    # B = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE))
    # I = np.identity(SAMPLE_SIZE)
    # alpha = 0.1

    # for m in range(M):
    #     B += np.linalg.inv(kernel_list[m,:,:] + alpha * I)

    # B *= M
    # return B

    return kernel
    

