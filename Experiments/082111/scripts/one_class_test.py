print __doc__
import numpy as np
import pylab as pl
import matplotlib.font_manager
from sklearn import svm

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
num_data = X.shape[0]
num_train = int(num_data*.7)
X_train = X[:num_train]
# Generate some regular novel observations
X_test = X[num_train:]



# fit the model
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(X_train)

y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

Z = np.array([np.array(l) for l in np.load(data_dir+feature_file_names[1]) if len(l) ==640])

z_pred = clf.predict(Z)
print "Z percentage is: %f", np.sum(z_pred < 0)/float(z_pred.shape[0])

for feature_file_name in feature_file_names[1:]:
    Z = np.array([np.array(l) for l in np.load(data_dir+feature_file_name) if len(l) ==640])
    z_pred = clf.predict(Z)
    print "Z percentage for %s is: %f" % (feature_file_name,np.sum(z_pred < 0)/float(z_pred.shape[0]))



y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size

n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size
# plot the line, the points, and the nearest vectors to the plane
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
pl.title("Novelty Detection")
pl.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=pl.cm.Blues_r)
a = pl.contour(xx, yy, Z, levels=[0], linewidths=2, colors=’red’)
pl.contourf(xx, yy, Z, levels=[0, Z.max()], colors=’orange’)
b1 = pl.scatter(X_train[:, 0], X_train[:, 1], c=’white’)
b2 = pl.scatter(X_test[:, 0], X_test[:, 1], c=’green’)
c = pl.scatter(X_outliers[:, 0], X_outliers[:, 1], c=’red’)
pl.axis(’tight’)
pl.xlim((-5, 5))
pl.ylim((-5, 5))
pl.legend([a.collections[0], b1, b2, c],
          ["learned frontier", "training observations",
           "new regular observations", "new abnormal observations"],
          loc="upper left",
          prop=matplotlib.font_manager.FontProperties(size=11))

pl.xlabel(
"error train: %d/200 ; errors novel regular: %d/20 ; " \
"errors novel abnormal: %d/20"
% (n_error_train, n_error_test, n_error_outliers))
pl.show()

