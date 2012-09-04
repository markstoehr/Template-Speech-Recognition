# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

def fast_patch_coding(np.ndarray[DTYPE_t,ndim=3] patches,
                      np.ndarray[DTYPE_t,ndim=2] S_coded,
                      np.ndarray[DTYPE_t,ndim=2] S_num_patches,
                      np.ndarray[np.uint16_t,ndim=1] patch_ids,
                      np.ndarray[np.uint16_t,ndim=1] all_patch_rows,
                      np.ndarray[np.uint16_t,ndim=1] all_patch_cols):
    """
    Assumed that all of S is set to an initial background value that will be ignored for
    computing the sums
    """
    cdef int num_freq = S_coded.shape[0]
    cdef int num_time = S_coded.shape[1]
    cdef int num_patches = patches.shape[0]
    cdef int num_features = patch_ids.shape[0]
    cdef int patch_height = patches.shape[1]
    cdef int patch_width = patches.shape[2]
    cdef Py_ssize_t S_freq_idx, S_time_idx, cur_freq_idx, cur_time_idx
    cdef Py_ssize_t feature_idx,patch_idx, patch_freq_idx, patch_time_idx
    for feature_idx in range(num_features):
        patch_idx =patch_ids[feature_idx]
        S_freq_idx = all_patch_rows[feature_idx]
        S_time_idx = all_patch_cols[feature_idx]
        for patch_freq_idx in range(patch_height):
            cur_freq_idx = S_freq_idx+patch_freq_idx
            for patch_time_idx in range(patch_width):
                cur_time_idx = S_time_idx+patch_time_idx
                if S_num_patches[cur_freq_idx,
                                 cur_time_idx] < 1.:
                    S_coded[cur_freq_idx,cur_time_idx] = 0.
                S_coded[cur_freq_idx,cur_time_idx] += patches[patch_idx,patch_freq_idx,patch_time_idx]
                S_num_patches[cur_freq_idx,cur_time_idx] += 1.

    for S_freq_idx in range(num_freq):
        for S_time_idx in range(num_time):
            if S_num_patches[S_freq_idx,S_time_idx] > 0:
                S_coded[S_freq_idx,S_time_idx] /= S_num_patches[S_freq_idx,S_time_idx]


