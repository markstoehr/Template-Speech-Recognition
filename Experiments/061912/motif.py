import numpy as np
from scipy.linalg import eigh
from scipy.linalg import circulant


def generate_all_shifts(atom, signal_atom_diff):
    return circulant(np.hstack((atom, np.zeros(signal_atom_diff)))).T[:signal_atom_diff+1]

class shifted_dictionary:
    """
    Shifted Dictionary class
    keeps track of the shifted versions of the dictionary atoms
    
    keeps a counter of which dictionary items have been set so far
    
    access the dictionary by using the D variable that is set in the
    initialization
    """
    def __init__(self,num_atoms, atom_length, signal_atom_diff):
        """
        Initialize the dictionary

        Indicate that we are going to fill the first dict atom
        """
        self.num_atoms = num_atoms
        self.atom_length = atom_length
        self.signal_atom_diff = signal_atom_diff
        # plus one is since this captures all the possible shifts,
        # beginning with the atoms on the far left of the vector
        # (towards the beginning of the signal) to when the atom is at
        # the end fo the signal (on the right of the vector)
        self.D = np.zeros((self.num_atoms,
                           self.signal_atom_diff+1,
                           self.atom_length + self.signal_atom_diff,
                           ))
        # indic
        self.cur_atom_id = -1
    #
    # access the dictionary
    def get_dict(self):
        """
        Calls the dictionary, only up to the atoms that have been set
        """
        return self.D[:self.cur_atom_id+1]
    #
    def get_shift_matcher(self):
        return 
    #
    #
    #
    def update_atom(self,atom,atom_id):
        self.cur_atom_id = max( atom_id, self.cur_atom_id)
        self.D[self.cur_atom_id] = generate_all_shifts(atom, 
                                                       self.signal_atom_diff)


def make_signal_shift_matcher(X,num_signals,signal_length,num_atoms,atom_size,num_shifts):
    """
    We need to match each shift of each atom to each signal, this means that we need
    a lot of copies of different signals

    This function takes our matrix of signals where each signal is a row
    for each atom we make a matrix with number of rows equal to the number of shifts
    
    number of shifts = signal_atom_diff + 1

    of a given signal copied, and we do that for each signal so that array has dimension
    num_signals \times num_shifts \times atom_size

    we do all the shifting and matching to the original signal since that way we don't have to do
    any shifting in our dictionary
    as that means everything
    is more compact when we are doing matching

    we repeat that above array for each atom
    """
    

def motif(X,num_atoms,tol,num_shifts=None):
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
    if num_shifts is None:
        num_shifts = X.shape[1]/4
    # Set the signal length and the number of signals
    num_signals, signal_length = X.shape
    # we start with just the middle of all the signals
    X_shifted = X[:,num_shifts/2:-num_shifts/2]
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
    shift_tracker = np.zeros((num_signals,signal_atom_diff))
    # initialize the shifted dictionary
    SD = shifted_dictionary(num_atoms, atom_length, signal_atom_diff)
    SD.update_atom(G[0],0)
    # matrix that we use to do the comparison with each of the shifted atoms
    # makes many copies of X
    #
    # Central to this is shift-matching that is for each shift and each atom
    # we have a copy of the original signals
    #
    #
    X_shift_matcher = np.tile(X.reshape(1,num_signals,signal_length),(num_atoms,1,1))
    for cur_atom in xrange(num_atoms):
        cur_error, prev_error = np.inf,-np.inf
        while cur_error - prev_error > tol:
            
            shift_tracker = np.argmax(
                np.dot(np.tile(X,),
                axis=1)
            for n in xrange(num_signals):
                shift_tracker np.argmax(n
        add_circ = circulant(np.hstack((cur_g,np.zeros(signal_atom_diff))))[:,:signal_atom_diff+1]
        B = np.dot(add_circ,add_circ.T)

        
        
        
    
