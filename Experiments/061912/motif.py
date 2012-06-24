import numpy as np
from scipy.linalg import eigh
from scipy.linalg import circulant


def generate_all_shifts(atom, signal_atom_diff):
    return circulant(np.hstack((atom, np.zeros(signal_atom_diff)))).T[:signal_atom_diff+1]


def get_shifted_X(X,num_signals,signal_length,atom_length,num_shifts):
    """
    We need to match each shift of each atom to each signal, this means that we need
    a lot of copies of different signals

    This function takes our matrix of signals where each signal is a row
    for each atom we make a matrix with number of rows equal to the number of shifts
    
    number of shifts = signal_atom_diff + 1

    of a given signal copied, and we do that for each signal so that array has dimension
    num_signals \times num_shifts \times atom_length

    we do all the shifting and matching to the original signal since
    that way we don't have to do any shifting in our dictionary as
    that means everything is more compact when we are doing matching
    """
    signal_shift_mat = np.zeros((num_signals,num_shifts,atom_length))
    for i in xrange(num_signals):
        signal_shift_mat[i] = circulant(X[i]).T[:num_shifts,-atom_length:]
    return signal_shift_mat

def set_atom_dict(atom, atom_id,atom_dict,
                  num_signals, num_shifts, atom_length):
    atom_dict[atom_id] = np.tile(atom.reshape(1,1,atom_length),
                                 (num_signals, num_shifts, 1))

def init_atom_dict(num_atoms,num_signals, num_shifts, atom_length):
    """
    Creates a large redundant array of the atoms we are using to decompose the signals
    such that we do fast correlation computation, and figure out which shift for each signal
    and each atom we should use
    """
    atom_dict = np.zeros((num_atoms,num_signals, num_shifts, atom_length))
    return atom_dict

def init_atom(atom_length):
    """
    Returns an initial atom drawn from a Gaussian distribution
    """
    return np.random.standard_normal(atom_length)

def match_X_shifted2atom_dict(X_shifted,atom,atom_length,num_signals,
                              num_shifts):
    """
    for the atom indexed by atom_id
    we compute which shift (in terms of the shift_id in the X_shifted signal
    gives the largest correlation for each signal in our original signal matrix
    the result is a list of the indices for each signal what the optimal is 
    """
    return np.argmax(np.abs(np.sum(X_shifted * np.tile(atom.reshape(1,1,atom_length),
                                 (num_signals, num_shifts, 1)),axis=2)),axis=1)

def get_X_best_shifts(X_shifted,best_shifts,num_signals):
    """
    best_shifts is computed from match_X_shifted2atom_dict
    X_shifted is going to have those best shifts so we can compute
    the eigen-decomposition

    X_shifted consists a copy of itself for each atom, and for each item
    we have a matrix for every signal in the original data matrix X
    each of these matrices is that original signal truncated and shifted
    to fit the atom sizes
    
    for each of those matrices that associates an atom with all shifts of a particular signal
    we have the index of the best fit, we wish to grab that one, and to do it efficiently
    """
    return X_shifted[np.arange(num_signals),best_shifts]
    

def compute_best_atom(X_shifted,g,B_cov, tol,
                      num_signals, signal_length, atom_length, num_shifts,
                      verbose=False,max_iters=np.inf):
    """
    X_shifted:
        Shifted version of all the signals
    g:
        initial guess for the next atom
    B_cov:
        covariance type matrix computed from the previous atoms,
        this is used because as we iteratively generate the items
        we take the next atom that does well on the residual of the signal
        Assumed to be symmetric
    tol:
        The tolerance for when we decide that the best approximate atom has been found
        looks at when the error between sucessive stages is sufficiently small
   
    Other parameters are:
    num_signals, signal_length, atom_length, num_shifts
    which should all be self-explanatory

    This function does two things:
    compute the optimal shift for the current guess of the 
    """
    g_diff_norm = np.inf
    num_iters = 0
    while g_diff_norm > tol and num_iters < max_iters:
        if verbose and num_iters % 50 == 0:
            print "On iteration %d in the dictionary update with error %g" % (num_iters, g_diff_norm)
        best_shifts= match_X_shifted2atom_dict(X_shifted,g,atom_length,num_signals,
                              num_shifts)
        X_best_shifts = get_X_best_shifts(X_shifted,best_shifts,num_signals)
        A = np.dot(X_best_shifts.T, X_best_shifts)
        _,cur_g=eigh(A,b=B_cov,eigvals_only =False, overwrite_a = True,eigvals=(0,0))
        g_diff =  g - cur_g[:,0]
        g_diff_norm = np.sqrt(np.dot(g_diff,g_diff))
        g = cur_g[:,0]
        num_iters += 1
    return g
        
def get_all_atom_shift_out_products(g,atom_length):
    all_atom_shifts = circulant(np.hstack((g, np.zeros(atom_length-1))))[:atom_length].T
    return np.dot(all_atom_shifts.T,all_atom_shifts)



def motif(X_orig,num_atoms,tol,num_shifts=None,
          verbose=False,max_iters=np.inf):
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
    # make the vector zero mean for all the signals
    X=X_orig - np.tile(np.mean(X_orig,axis=1),(X_orig.shape[1],1)).T
    if num_shifts is None:
        num_shifts = X.shape[1]/4
    # Set the signal length and the number of signals
    num_signals, signal_length = X.shape
    # we start with just the middle of all the signals
    atom_length = signal_length - num_shifts+1
    # difference between signal length and atom length
    signal_atom_diff = signal_length-atom_length
    atom_id = 0
    # create the matrix of atoms
    G = np.zeros((num_atoms,atom_length))
    X_shifted = get_shifted_X(X,num_signals,signal_length,atom_length,num_shifts)
    g = init_atom(atom_length)
    # initialize the B_covariance to the identity
    B_cov = np.eye(atom_length)
    G[atom_id] = compute_best_atom(X_shifted,g,B_cov, tol,
                                   num_signals, signal_length, atom_length, num_shifts,
                                   verbose=verbose,max_iters = max_iters)
    B_cov += get_all_atom_shift_out_products(G[atom_id],atom_length)
    for atom_id in xrange(1,num_atoms):
        if verbose:
            print "construct atom %d" % atom_id
        g = init_atom(atom_length)
        G[atom_id] = compute_best_atom(X_shifted,g,B_cov, tol,
                      num_signals, signal_length, atom_length, num_shifts,
                                       verbose=verbose, max_iters = max_iters)
        B_cov += get_all_atom_shift_out_products(G[atom_id],atom_length)
    return G

        
        
    
