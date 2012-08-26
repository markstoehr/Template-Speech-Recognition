print __doc__
import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn.preprocessing import Scaler
from sklearn.datasets import load_iris
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold


import cPickle, os
root_dir = '/home/mark/Template-Speech-Recognition/'
data_dir = root_dir + 'Experiments/081112/data/'

feature_file_suffix = 'feature_list.npy'

feature_file_names = [
    feature_file_name
    for feature_file_name in os.listdir(data_dir)
    if feature_file_name[-len(feature_file_suffix):] == feature_file_suffix]

feature_file_idx = 0
# Generate train data
X = np.array([np.array(l) for l in np.load(data_dir+feature_file_names[feature_file_idx]) if len(l) ==640])

num_train_data = X.shape[0]

max_examples_per_class = 100
Y = np.zeros((0,640))
for cur_idx, feature_file_name in enumerate(feature_file_names[1:]):
    if cur_idx % 20 == 0:
        print cur_idx
    Z = np.array([np.array(l) for l in np.load(data_dir+feature_file_name) if len(l) ==640])
    if len(Z) == 0: continue
    Y = np.vstack(
        (Y,
         Z[:100]))




scaler = Scaler()

num_true_class = X.shape[0]
num_false_class = Y.shape[0]

X_true = X.copy()
X_false = Y.copy()

X = np.vstack((X_true,X_false))

Y = np.zeros(X.shape[0])
Y[num_true_class:] = 1

X = scaler.fit_transform(X)
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
C_range = 10. ** np.arange(-2, 9)
gamma_range = 10. ** np.arange(-5, 4)
param_grid = dict(gamma=gamma_range, C=C_range)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=Y, k=5))
grid.fit(X, Y)
print("The best classifier is: ", grid.best_estimator_)
