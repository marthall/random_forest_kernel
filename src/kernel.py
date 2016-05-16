import pandas as pd
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestClassifier
from itertools import compress
from func import get_leaves
from func import get_lineage

def get_partial_kernel(forest, dataset, labels):
    SAMPLE_SIZE = len(dataset)
    # print "Sample size: {}".format(SAMPLE_SIZE)

    TreeIndex = randint(0,forest.n_estimators-1)
    Arbre = forest.estimators_[TreeIndex]
    n_nodes = Arbre.tree_.node_count
    children_left = Arbre.tree_.children_left
    children_right = Arbre.tree_.children_right

    # h =Arbre.tree_.max_depth #all trees don't have the same height???

    #Node selction
    leaf_index = Arbre.apply(dataset)
    leaf_nodes = np.array(list(set(leaf_index)))  # Remove duplicates
    # print "Leaf index size: {}".format(len(leaf_nodes))

    is_leaves, node_depth = get_leaves(children_left, children_right, n_nodes)
    # print "is_leaves: {}".format(sum(is_leaves))
    myparents = get_lineage(Arbre, leaf_index, node_depth)

    myparents = np.array(myparents)

    h = min(node_depth[is_leaves]) - 1
    d = randint(0, h)

    # print "Max-height: {}".format(h)
    # print "Choses-height: {}".format(d)

    nth_parent_for_sample = []
    for lineage in myparents:
        try:
            nth_parent_for_sample.append(lineage[d])
        except IndexError:
            print "Error!!!"
            print lineage

    partial_kernel = np.zeros([SAMPLE_SIZE, SAMPLE_SIZE])

    for i in range(SAMPLE_SIZE):
        for j in range(i, SAMPLE_SIZE):
            if nth_parent_for_sample[i][0] == nth_parent_for_sample[j][0]:
                partial_kernel[i][j] = 1
                partial_kernel[j][i] = 1

    return partial_kernel


def get_kernel(dataset, labels):

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(dataset, labels)

    SAMPLE_SIZE = len(dataset)
    M = 10

    kernel_list = np.empty([M, SAMPLE_SIZE, SAMPLE_SIZE])
    for m in range(M):
        print "Building partial kernel: {}".format(m)
        kernel_list[m,:,:] = get_partial_kernel(forest, dataset, labels)

    kernel = np.mean(kernel_list, axis=0)
    print kernel
    return kernel

