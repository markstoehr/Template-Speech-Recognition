#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def subtract_max_affinities(np.ndarray[DTYPE_t, ndim=3] affinities, 
                            np.ndarray[DTYPE_t,ndim=1] max_affinities,
                            int num_mix,
                            int num_data,
                            int trans_amount):
    cdef unsigned int mix_idx = 0
    cdef unsigned  int mix_stride = num_data * trans_amount
    cdef unsigned  int data_idx = 0
    cdef unsigned  int data_stride = trans_amount
    cdef unsigned  int trans_idx = 0
    
    for mix_idx in range(num_mix):
        for data_idx in range(num_data):
            for trans_idx in range(trans_amount):
                affinities[mix_idx,data_idx,trans_idx] -= max_affinities[data_idx]

def normalize_affinities(np.ndarray[DTYPE_t, ndim=3] affinities, 
                            np.ndarray[DTYPE_t,ndim=1] affinity_sums,
                            int num_mix,
                            int num_data,
                            int trans_amount):
    cdef unsigned int mix_idx = 0
    cdef unsigned  int mix_stride = num_data * trans_amount
    cdef unsigned  int data_idx = 0
    cdef unsigned  int data_stride = trans_amount
    cdef unsigned  int trans_idx = 0
    
    for mix_idx in range(num_mix):
        for data_idx in range(num_data):
            for trans_idx in range(trans_amount):
                affinities[mix_idx,data_idx,trans_idx] /= affinity_sums[data_idx]

def get_marginalized_transformations
    
