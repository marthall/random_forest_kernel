from __future__ import print_function
"""
======================
SVM with custom kernel
======================

Simple usage of Support Vector Machines to classify a sample. It will
plot the decision surface and the support vectors.

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from datasets import wine, adult, isolet

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

from kernel import get_kernel




dataset = "iris"

# Available datasets:
# - iris
# - diabetes
# - wine
# - adult


if dataset == "iris":
  # import some data to play with
  iris = datasets.load_iris()
  X = iris.data[:, :2]  # we only take the first two features. We could
                        # avoid this ugly slicing by using a two-dim dataset
  Y = iris.target

elif dataset == "digits":
  # import some data to play with
  data = datasets.load_digits()
  X = data.data[:, :2]  # we only take the first two features. We could
                        # avoid this ugly slicing by using a two-dim dataset
  Y = data.target

elif dataset == "wine":
  X, Y = wine.load_dataset()

elif dataset == "adult":
  X, Y = adult.load_dataset()

elif dataset == "isolet":
  X, Y = isolet.load_dataset()


# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.5, random_state=0)

kernel = get_kernel(X_train, y_train)

def my_kernel(X, Y):
    return kernel





# # Set the parameters by cross-validation
# tuned_parameters = [{
#     'kernel': ['rbf'],
#     'gamma': [1e-3, 1e-4],
#     'C': [1, 10, 100, 1000]
#     },{
#     'kernel': ['poly'],
#     'gamma': [1e-3, 1e-4],
#     'C': [1, 10, 100, 1000],
#     'degree': [1,2,3,4]
# }]

# score = 'precision'

# print("# Tuning hyper-parameters for %s" % score)
# print()

# clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=10,
#                    scoring='%s_weighted' % score)
# clf.fit(X_train, y_train)

# print("Best parameters set found on development set:")
# print()
# print(clf.best_params_)
# print()
# print("Grid scores on development set:")
# print()
# for params, mean_score, scores in clf.grid_scores_:
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean_score, scores.std() * 2, params))
# print()

# print("Detailed classification report:")
# print()
# print("The model is trained on the full development set.")
# print("The scores are computed on the full evaluation set.")
# print()
# y_true, y_pred = y_test, clf.predict(X_test)
# print(classification_report(y_true, y_pred))
# print()


# C = clf.best_params_["C"]
# gamma = clf.best_params_["gamma"]
# # degree = clf.best_params_["degree"]
# degree = 3
# kernel = clf.best_params_["kernel"]




# we create an instance of SVM and fit out data.
svm_c = SVC(kernel=my_kernel)
# svm_c = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
svm_c.fit(X_train, y_train)
# print("Hei######################################################")
print(svm_c.score(X_test, y_test))
# print(svm_c.predict(X_test))

svm_c = SVC(kernel="rbf")
# svm_c = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
svm_c.fit(X_train, y_train)
print(svm_c.score(X_test, y_test))




# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

# h = .02  # step size in the mesh

# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = svm_c.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
# plt.title('3-Class classification using Support Vector Machine with custom'
#           ' kernel')
# plt.axis('tight')
# plt.show()



# if __name__ == "__main__":

#   datasets = ["iris", "wine"]
#   methods = ["svm_c", "svm_nu"]



