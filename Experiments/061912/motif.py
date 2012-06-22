import numpy as np
from scipy.linalg import eigh
from scipy.linalg import circulant


def motif(X,dict_size,tol,shift_range=None):
    """
    X is the set of training signals where each row
    corresponds to a signal and the columns are time points in the signal
    dict_size is the number of atoms to learn in the signal

    shift_range specifies how much shifting is allowed for the atoms
    by default its half the signal length
    """
    #
    # first we do the initialization
    # here the shifts don't matter at all
    # 
    if shift_range is None:
        shift_range = X.shape[1]/4
    # Set the signal length and the number of signals
    num_signals, signal_length = X.shape
    # we start with just the middle of all the signals
    X_shifted = X[:,shift_range:-shift_range]
    # compute the gram matrix
    A = np.dot(X_shifted.T, X_shifted)
    # eigendecomposition
    _,cur_g=eigh(A,eigvals_only =False, overwrite_a = True,eigvals=(0,0))
    atom_size = X_shifted.shape[1]
    # difference between signal length and atom length
    signal_atom_diff = signal_length-atom_size
    G = np.zeros((dict_size,atom_size))
    G[0] = cur_g[:,0]
    B = np.zeros((signal_length, signal_length))
    # keep track of what the best shifts are
    shift_tracker = np.zeros((num_signals,signal_atom_diff
    for i in xrange(dict_size):
        cur_error = np.inf
        while cur_error > tol:
            for n in xrange(num_signals):
                shift_tracker
        add_circ = circulant(np.hstack((cur_g,np.zeros(signal_atom_diff))))[:,:signal_atom_diff+1]
        B = np.dot(add_circ,add_circ.T)

        
        
        
    
