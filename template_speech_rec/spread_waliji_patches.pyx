#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float32
UINT = np.uint8
ctypedef np.float32_t DTYPE_t

ctypedef np.uint8_t UINT_t

def spread_waliji_patches(np.ndarray[ndim=2,dtype=np.int64_t] X,
                          int spread_0_dim,
                          int spread_1_dim,
                          int num_parts):
    """
    Parameters:
    ===========
    X: np.ndarray[ndim=2,dtype=int]
        Best feature fit for the different locations on the 
        data.  A feature of value zero means that there were
        insufficient edges in that region
        a feature value in [1 ... num_parts] means that 
        a feature was a best fit there.
    spread_0_dim: int
        Radius of this size indicates the spread region along
        the 0th axis.  0 corresponds to no spread. 1 Corresponds
        to spreading over 1 cell in both directions (boundaries
        are handled by assuming zeros in all other coordinates)
    spread_1_dim: int
        Radius of the spread along dimension 1
    Output:
    =======
    bin_out_map: np.ndarray[ndim=3,dtype=np.uint8]
        Performs spreading 
    """
    cdef np.uint16_t X_dim_0 = X.shape[0]
    cdef np.uint16_t X_dim_1 = X.shape[1]
    cdef np.ndarray[ndim = 3, dtype=UINT_t] bin_out_map = np.zeros((X_dim_0,
                                                                    X_dim_1,
                                                                    num_parts),
                                                                   dtype=np.uint8)
    cdef int i,j,lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx,x0,x1
    for i in range(X_dim_0):
        lo_spread_0_idx = max(i-spread_0_dim,0)
        hi_spread_0_idx = min(i+spread_0_dim+1,X_dim_0)
        for j in range(X_dim_1):
            lo_spread_1_idx = max(j-spread_1_dim,0)
            hi_spread_1_idx = min(j+spread_1_dim+1,X_dim_1)
            for x0 in range(lo_spread_0_idx,hi_spread_0_idx):
                for x1 in range(lo_spread_1_idx,hi_spread_1_idx):
                    if X[x0,x1] > 0:
                        bin_out_map[i,j,X[x0,x1]-1] = 1
                    else:
                        pass
    return bin_out_map
