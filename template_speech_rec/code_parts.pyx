# cython code_parts.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#      -I/usr/include/python2.7 -o code_parts.so code_parts.c
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float64
UINT = np.uint8
ctypedef np.float64_t DTYPE_t

ctypedef np.uint8_t UINT_t

cdef _count_edges(np.ndarray[ndim=3,dtype=UINT_t] X,
                 unsigned int i_start,
                 unsigned int i_end,
                 unsigned int j_start,
                 unsigned int j_end,
                 unsigned int num_z):
    cdef unsigned int count = 0
    cdef unsigned int i,j,z
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            for z in range(num_z):
                if X[i,j,z]:
                    count += 1
    return count

cdef _count_edges_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
                      np.ndarray[ndim=2,dtype=UINT_t] M,
                 unsigned int i_start,
                 unsigned int i_end,
                 unsigned int j_start,
                 unsigned int j_end,
                 unsigned int num_z):
    cdef unsigned int count = 0
    cdef unsigned int i,j,z
    for i in range(i_start,i_end):
        for j in range(j_start,j_end):
            if M[i,j]:
                for z in range(num_z):
                    if X[i,j,z]:
                        count += 1
    return count


# cdef compute_loglikelihoods(np.ndarray[ndim=2,dtype=UINT_t] X,
#                            unsigned int i_start,
#                            unsigned int i_end,
#                            unsigned int j_start,
#                            unsigned int j_end,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] log_parts,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] log_invparts,
#                            np.ndarray[ndim=3,dtype=DTYPE_t] out_map,
#                            unsigned int num_parts):
#     for i in range(i_end-i_start):
#         for j in range(j_end-j_start):
#             if X[i_start+i,j_start+j]:
#                 for k in range(num_parts):
#                     out_map[i_start,j_start,k] += log_parts[k,i,j]
#             else:
#                 for k in range(num_parts):
#                     out_map[i_start,j_start,k] += log_invparts[k,i,j]


def code_parts(np.ndarray[ndim=3,dtype=UINT_t] X,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
               np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
               int threshold):
    """
    At each location of `X`, find the log probabilities for each part and location. Outputs these part assignments in the same data dimensions as `X`. Neighborhoods of `X` with edge counts lower than `threshold` are regarded as background and assigned zero.

    Parameters
    ----------
    X : ndarray[ndim=3,dtype=np.uint8]
        The first two dimensions of the array specify locations. The last one specifies a binary edge type. The value ``X[s,t,e]`` is 1 iff there is an edge of type `e` detected at location `(s,t)`.
    log_parts : ndarray[ndim=4]
        We have a Bernoulli mixture model defined over patches of the input image. The `log_parts` is a logarithm applied to the array of edge probability maps for each part. Array of shape `(K, S, T, E)`, where `K` is the number of mixture component, `S` and `T` the shape of the data, and `E` the number of edges. The value of ``log_parts[k,s,t,e]`` is the log probability of observing an edge `e` at location `(s,t)`, conditioned on the mixture component being `k`.
    log_invparts : ndarray[ndim=4]
        Preprocessed inverse of `log_parts`, i.e. ``log(1-exp(log_parts))``.
    threshold : int
        The least number of edges in a patch to reject the null background hypothesis.
    
    Returns
    -------
    out_map : ndarray[ndim=3] 
        Array of shape `(S, T, K+1)`. There are two cases, either the third dimension is `(0, -inf, -inf, ...)`, when there are insufficient edges in the neighborhood of a location. Otherwise, `out_map[s,t,i+1]` is the log likelihood of part `i` at location `(s,t)`. Additionally, `out_map[s,t,0]` is equal to `-inf`.
    """

    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = -np.inf * np.ones((new_x_dim,
                                                                        new_y_dim,
                                                                        num_parts+1),dtype=DTYPE)
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim
            count = _count_edges(X,i_start,i_end,j_start,j_end,X_z_dim)
            if count >= threshold:
                out_map[i_start,j_start] = 1.0
                out_map[i_start,j_start,0] = -np.inf
                for i in range(part_x_dim):
                    for j in range(part_y_dim):
                        for z in range(X_z_dim):
                            if X[i_start+i,j_start+j,z]:
                                for k in range(num_parts):
                                    out_map[i_start,j_start,k+1] += log_parts[k,i,j,z]
                            else:
                                for k in range(num_parts):
                                    out_map[i_start,j_start,k+1] += log_invparts[k,i,j,z]
            else:
                out_map[i_start,j_start,0] = 0.0
                
    return out_map



def code_parts_fast(np.ndarray[ndim=3,dtype=UINT_t] X,
                    np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
                    np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int threshold):
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,i,j,z,k
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = np.zeros((new_x_dim,
                                                        new_y_dim,
                                                                   num_parts+1),dtype=DTYPE)
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen
    
    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim
            count = _count_edges(X,i_start,i_end,j_start,j_end,X_z_dim)
            if count >= threshold:
                for i in range(part_x_dim):
                    for j in range(part_y_dim):
                        for z in range(X_z_dim):
                            if X[i_start+i,j_start+j,z]:
                                for k in range(num_parts):
                                    out_map[i_start,j_start,k+1] += log_parts[k,i,j,z]
                            else:
                                for k in range(num_parts):
                                    out_map[i_start,j_start,k+1] += log_invparts[k,i,j,z]
                                    
                # every term is a log-likelihood so adding them up will be bounded by the smallest log-likelihood
                for k in range(num_parts):
                    out_map[i_start,j_start,0] += out_map[i_start,j_start,k+1] - 1
            else:
                out_map[i_start,j_start,0] = 1.
                
    return out_map

def get_parts_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
                   np.ndarray[ndim=2,dtype=UINT_t] M,
                   np.ndarray[ndim=2,dtype=DTYPE_t] S,
                   int threshold):
    """
    S is the spectrogram we return those patches as well
    """
    cdef unsigned int part_x_dim = M.shape[0]
    cdef unsigned int part_y_dim = M.shape[1]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int num_patches = new_x_dim * new_y_dim
    cdef unsigned int i0,j0,patch_id,i,j,z,k,num_above_thresh
    cdef np.ndarray[ndim=2,dtype=np.uint32_t] patch_counts = np.zeros(
        (new_x_dim,new_y_dim),dtype=np.uint32)
    num_above_thresh = 0
    for i in range(new_x_dim):
        for j in range(new_y_dim):
            patch_counts[i,j] = _count_edges_mask(X,M,i,i + part_x_dim,
                                                 j,j+part_y_dim,X_z_dim)
            if patch_counts[i,j] >= threshold:
                num_above_thresh += 1
    
    # now we allocate the array that will have the patches
    cdef np.ndarray[ndim=4,dtype=UINT_t] patch_set = np.zeros(
        (num_above_thresh,part_x_dim,part_y_dim,X_z_dim),
        dtype=np.uint8)
    cdef np.ndarray[ndim=3,dtype=DTYPE_t] S_patch_set = np.zeros(
        (num_above_thresh,part_x_dim,part_y_dim),
        dtype=DTYPE)
    cdef np.ndarray[ndim=2,dtype=np.uint32_t] patch_locs = np.zeros(
        (num_above_thresh,2),dtype=np.uint32)

    patch_id = 0
    for i in range(new_x_dim):
        for j in range(new_y_dim):
            if patch_counts[i,j] >= threshold:
                patch_locs[patch_id,0] = i
                patch_locs[patch_id,1] = j
                for i0 in range(part_x_dim):
                    for j0 in range(part_y_dim):
                        S_patch_set[patch_id,
                                    i0,
                                    j0] = S[i+i0,j+j0]
                        for z in range(X_z_dim):
                            patch_set[patch_id,
                                      i0,j0,z] = X[i+i0,j+j0,z]
                patch_id += 1
    return patch_set, S_patch_set, patch_counts, patch_locs
                
            
            

def code_parts_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
                    np.ndarray[ndim=2,dtype=UINT_t] M,
                    np.ndarray[ndim=4,dtype=DTYPE_t] log_parts,
                    np.ndarray[ndim=4,dtype=DTYPE_t] log_invparts,
                    int mask_threshold,
                    int absolute_threshold):
    cdef unsigned int num_parts = log_parts.shape[0]
    cdef unsigned int part_x_dim = log_parts.shape[1]
    cdef unsigned int part_y_dim = log_parts.shape[2]
    cdef unsigned int part_z_dim = log_parts.shape[3]
    cdef unsigned int X_x_dim = X.shape[0]
    cdef unsigned int X_y_dim = X.shape[1]
    cdef unsigned int X_z_dim = X.shape[2]
    cdef unsigned int new_x_dim = X_x_dim - part_x_dim + 1
    cdef unsigned int new_y_dim = X_y_dim - part_y_dim + 1
    cdef unsigned int i_start,j_start,i_end,j_end,count,mask_count,i,j,z,k
    # we have num_parts + 1 because we are also including some regions as being
    # thresholded due to there not being enough edges
    

    cdef np.ndarray[dtype=DTYPE_t, ndim=3] out_map = np.zeros((new_x_dim,
                                                        new_y_dim,
                                                                   num_parts+1),dtype=DTYPE)
    # The first cell along the num_parts+1 axis contains a value that is either 0
    # if the area is deemed to have too few edges or min_val if there are sufficiently many
    # edges, min_val is just meant to be less than the value of the other cells
    # so when we pick the most likely part it won't be chosen

    for i_start in range(X_x_dim-part_x_dim+1):
        i_end = i_start + part_x_dim
        for j_start in range(X_y_dim-part_y_dim+1):
            j_end = j_start + part_y_dim
            count = _count_edges(X,i_start,i_end,j_start,j_end,X_z_dim)
            mask_count = _count_edges_mask(X,M,i_start,i_end,j_start,j_end,X_z_dim)
            if (count >= absolute_threshold) and (mask_count > mask_threshold):
                for i in range(part_x_dim):
                    for j in range(part_y_dim):
                        for z in range(X_z_dim):
                            if X[i_start+i,j_start+j,z]:
                                for k in range(num_parts):
                                    out_map[i_start,j_start,k+1] += log_parts[k,i,j,z]
                            else:
                                for k in range(num_parts):
                                    out_map[i_start,j_start,k+1] += log_invparts[k,i,j,z]
                                    
                # every term is a log-likelihood so adding them up will be bounded by the smallest log-likelihood
                for k in range(num_parts):
                    out_map[i_start,j_start,0] += out_map[i_start,j_start,k+1] - 1
            else:
                out_map[i_start,j_start,0] = 1.
                
    return out_map

def spread_patches(np.ndarray[ndim=2,dtype=np.int64_t] X,
                   int spread_0_dim,
                   int spread_1_dim,
                   int num_parts):
    """
    Performs patch spreading according to Bernstein and Amit [1].

    Parameters
    ----------
    X : ndarray[ndim=2,dtype=int]
        Best feature fit for the different locations on the 
        data.  A feature of value zero means that there were
        insufficient edges in that region
        a feature value in [1 ... num_parts] means that 
        a feature was a best fit there.
    spread_0_dim : int
        Radius of this size indicates the spread region along
        the 0th axis.  0 corresponds to no spread. 1 Corresponds
        to spreading over 1 cell in both directions (boundaries
        are handled by assuming zeros in all other coordinates)
    spread_1_dim : int
        Radius of the spread along dimension 1
    num_parts : int
        Number of parts.

    Returns
    -------
    bin_out_map : np.ndarray[ndim=3,dtype=np.uint8]
        Performs spreading and returns TODO.

    References
    ----------
    [1] E.J. Bernstein, Y. Amit : Part-Based Statistical Models for Object Classification and Detection (2005)
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
    return bin_out_map
