import numpy as np
from scipy.linalg import eigh
from scipy.linalg import circulant


# want to see if the motif algorithm as written will pick up on sinusoidal structure
# test matrix should be shifted versions of a sinuisoid
s=np.sin(np.arange(100)/99. * 10.*np.pi)
X = circulant(np.hstack((s,np.zeros(30))))[:,:30].T

assert X.shape[0] == 30


for i in xrange(30):
    np.testing.assert_array_almost_equal(X[i][i:s.shape[0]+i],s)

dict_size = 4
shift_range = 15
tol = .0001

if shift_range is None:
        shift_range = X.shape[1]/4



num_signals, signal_length = X.shape
X_shifted = X[:,shift_range:-shift_range]
A = np.dot(X_shifted.T, X_shifted)
_,cur_g=eigh(A,eigvals_only =False, overwrite_a = True,eigvals=(0,0))
atom_size = X_shifted.shape[1]
signal_atom_diff = signal_length-atom_size
G = np.zeros((dict_size,atom_size))
G[0] = cur_g[:,0]
B = np.zeros((signal_length, signal_length))
# keep track of what the best shifts are
shift_tracker = np.zeros((num_signals,signal_atom_diff))

G.shape
