import numpy as np
from scipy.linalg import eigh
from scipy.linalg import circulant


# want to see if the motif algorithm as written will pick up on sinusoidal structure
# test matrix should be shifted versions of a sinuisoid
s=np.sin(np.arange(100)/99. * 10.*np.pi)


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
    #
    #
    def update_atom(self,atom,atom_id=None):
        self.cur_atom_id = max( atom_id, self.cur_atom_id)
        self.D[self.cur_atom_id] = generate_all_shifts(atom, 
                                                       self.signal_atom_diff)



num_atoms = 4
atom_length = 3
signal_atom_diff = 2
SD = shifted_dictionary(num_atoms, atom_length, signal_atom_diff)


assert np.all(generate_all_shifts(np.array([2,3,4]),2) == np.array([[2,3,4,0,0],[0,2,3,4,0],[0,0,2,3,4]]))

X = generate_all_shifts(s,30)

assert X.shape[0] == 31


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
