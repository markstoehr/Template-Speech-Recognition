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
DTYPE = np.float32
UINT = np.uint8
ctypedef np.float32_t DTYPE_t

ctypedef np.uint8_t UINT_t

cdef count_edges(np.ndarray[ndim=3,dtype=UINT_t] X,
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

cdef count_edges_mask(np.ndarray[ndim=3,dtype=UINT_t] X,
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
            count = count_edges(X,i_start,i_end,j_start,j_end,X_z_dim)
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
            patch_counts[i,j] = count_edges_mask(X,M,i,i + part_x_dim,
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
            count = count_edges(X,i_start,i_end,j_start,j_end,X_z_dim)
            mask_count = count_edges_mask(X,M,i_start,i_end,j_start,j_end,X_z_dim)
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

