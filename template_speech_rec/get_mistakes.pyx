# cython get_mistakes.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#      -I/usr/include/python2.7 -o get_mistakes.so get_mistakes.c
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float32
UINT = np.uint16
ctypedef np.float32_t DTYPE_t

ctypedef np.uint16_t UINT_t
ctypedef np.uint8_t BOOL_t
ctypedef np.int16_t INT_t

def get_example_scores_metadata(np.ndarray[ndim=2,dtype=BOOL_t] example_mask,
                 np.ndarray[ndim=2,dtype=UINT_t] classify_locs,
                 np.ndarray[ndim=2,dtype=DTYPE_t] classify_array,
                 np.ndarray[ndim=1,dtype=UINT_t] classify_lengths,
                 int num_examples):
    cdef int num_utts = example_mask.shape[0]
    cdef int max_utt_len = example_mask.shape[1]
    cdef int utt_id,phn_id,cur_example_id
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] example_scores = np.zeros(num_examples,
                                                                    dtype=DTYPE)

    cdef np.ndarray[ndim=2,dtype=UINT_t] example_metadata = np.zeros(
        (num_examples,
         3),
        dtype=UINT)
    cur_example_id = 0
    for utt_id in range(num_utts):
        for phn_id in range(classify_lengths[utt_id]):
            if example_mask[utt_id,phn_id]:
                example_scores[cur_example_id] = classify_array[utt_id,phn_id]
                example_metadata[cur_example_id,0] = utt_id
                example_metadata[cur_example_id,1] = phn_id
                example_metadata[cur_example_id,2] = classify_locs[utt_id,phn_id]
                cur_example_id += 1
    return example_scores,example_metadata
