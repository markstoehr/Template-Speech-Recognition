# cython compute_likelihood_linear_filter.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#      -I/usr/include/python2.7 -o compute_likelihood_linear_filter.so compute_likelihood_linear_filter.c
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
cdef inline int min(int a, int b): return a if a <= b else b
cdef inline int max(int a, int b): return a if a >= b else b


def get_max_detect_examples(np.ndarray[ndim=4,
                               dtype=UINT_t] F,
                            np.ndarray[ndim=3,
                                       dtype=np.float64_t] LF,
                            np.ndarray[ndim=1,
                                       dtype=np.int64_t] lengths):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=4,dtype=UINT_t]
        Features for the example utterances that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F[i] and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    lengths: np.ndarray[ndim=1,dtype=np.int64_t]
        how far to go with detection of the detectors
    Output:
    =======
    detect_scores: np.ndarray[ndim=2,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t num_examples = F.shape[0]
    cdef np.uint32_t F_dim_0 = F.shape[1]
    cdef np.uint16_t F_dim_1 = F.shape[2]
    cdef np.uint16_t F_dim_2 = F.shape[3]
    cdef np.uint32_t LF_dim_0 = LF.shape[0]
    cdef np.ndarray[ndim=1,dtype=np.float64_t] max_detect_scores = np.zeros(num_examples,
                                                                   dtype=np.float64)
    cdef np.uint32_t cur_example,cur_detection,filter_timepoint,frequency,part_identity, max_time
    cdef np.float64_t cur_example_val
    for cur_example in range(num_examples):
        # cur_detection is the time point in the overall vector
        for cur_detection in range(lengths[cur_example]):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
            cur_example_val = 0.0
            max_time = min( F_dim_0 - cur_detection,
                            LF_dim_0)
            for filter_timepoint in range(max_time):
                for frequency in range(F_dim_1):
                    for part_identity in range(F_dim_2):
                        if F[cur_example,
                             cur_detection+filter_timepoint,
                             frequency,
                             part_identity]:
                            cur_example_val += (
                        
                                LF[filter_timepoint,
                                   frequency,
                                   part_identity])
            
            if cur_detection == 0 or cur_example_val > max_detect_scores[cur_example]:
                max_detect_scores[cur_example] = cur_example_val

    return max_detect_scores


def detect_examples(np.ndarray[ndim=4,
                               dtype=UINT_t] F,
                    np.ndarray[ndim=3,
                               dtype=np.float64_t] LF,
                    np.ndarray[ndim=1,
                               dtype=np.int64_t] lengths,
                    np.int64_t max_length):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=4,dtype=UINT_t]
        Features for the example utterances that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F[i] and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    lengths: np.ndarray[ndim=1,dtype=np.int64_t]
        how far to go with detection of the detectors
    Output:
    =======
    detect_scores: np.ndarray[ndim=2,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t num_examples = F.shape[0]
    cdef np.uint32_t F_dim_0 = F.shape[1]
    cdef np.uint16_t F_dim_1 = F.shape[2]
    cdef np.uint16_t F_dim_2 = F.shape[3]
    cdef np.uint32_t LF_dim_0 = LF.shape[0]
    cdef np.ndarray[ndim=2,dtype=np.float64_t] detect_scores = np.zeros((num_examples,
                                                                    max_length),
                                                                   dtype=DTYPE)
    cdef np.uint32_t cur_example,cur_detection,filter_timepoint,frequency,part_identity, max_time
    for cur_example in range(num_examples):
        # cur_detection is the time point in the overall vector
        for cur_detection in range(lengths[cur_example]):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
            max_time = min( F_dim_0 - cur_detection,
                            LF_dim_0)
            for filter_timepoint in range(max_time):
                for frequency in range(F_dim_1):
                    for part_identity in range(F_dim_2):
                        if F[cur_example,
                             cur_detection+filter_timepoint,
                             frequency,
                             part_identity]:
                            detect_scores[cur_example,
                                          cur_detection] += (
                        
                                LF[filter_timepoint,
                                   frequency,
                                   part_identity])
    return detect_scores



def detect(np.ndarray[ndim=3,
                      dtype=UINT_t] F,
           np.ndarray[ndim=3,
                      dtype=DTYPE_t] LF):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=3,dtype=UINT_t]
        Features for an example utterance that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    Output:
    =======
    detect_scores: np.ndarray[ndim=1,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t F_dim_0 = F.shape[0]
    cdef np.uint16_t F_dim_1 = F.shape[1]
    cdef np.uint16_t F_dim_2 = F.shape[2]
    cdef np.uint16_t LF_dim_0 = LF.shape[0]
    cdef np.uint16_t num_detections = F_dim_0 - LF_dim_0 +1
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] detect_scores = np.zeros(F_dim_0,
                                                                   dtype=DTYPE)
    cdef np.uint32_t cur_detection,filter_timepoint,frequency,part_identity,lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx,x0,x1
    # cur_detection is the time point in the overall vector
    for cur_detection in range(num_detections):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
        for filter_timepoint in range(LF_dim_0):
            for frequency in range(F_dim_1):
                for part_identity in range(F_dim_2):
                    if F[cur_detection+filter_timepoint,
                         frequency,
                         part_identity]:
                        detect_scores[cur_detection] += (
                        
                            LF[filter_timepoint,
                               frequency,
                               part_identity])
    return detect_scores

def detect_float(np.ndarray[ndim=3,
                      dtype=DTYPE_t] F,
           np.ndarray[ndim=3,
                      dtype=DTYPE_t] LF):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=3,dtype=UINT_t]
        Features for an example utterance that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    Output:
    =======
    detect_scores: np.ndarray[ndim=1,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t F_dim_0 = F.shape[0]
    cdef np.uint16_t F_dim_1 = F.shape[1]
    cdef np.uint16_t F_dim_2 = F.shape[2]
    cdef np.uint16_t LF_dim_0 = LF.shape[0]
    cdef np.uint16_t num_detections = F_dim_0 - LF_dim_0 +1
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] detect_scores = np.zeros(F_dim_0,
                                                                   dtype=DTYPE)
    cdef np.uint32_t cur_detection,filter_timepoint,frequency,part_identity,lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx,x0,x1
    # cur_detection is the time point in the overall vector
    for cur_detection in range(num_detections):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
        for filter_timepoint in range(LF_dim_0):
            for frequency in range(F_dim_1):
                for part_identity in range(F_dim_2):
                    detect_scores[cur_detection] += (
                        
                        LF[filter_timepoint,
                           frequency,
                           part_identity] 
                        * F[cur_detection+filter_timepoint,
                            frequency,
                            part_identity])
    return detect_scores

def detect_float_max(np.ndarray[ndim=3,
                      dtype=DTYPE_t] F,
           np.ndarray[ndim=4,
                      dtype=DTYPE_t] LF,
                     np.ndarray[ndim=1,dtype=DTYPE_t] cs):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=3,dtype=UINT_t]
        Features for an example utterance that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    Output:
    =======
    detect_scores: np.ndarray[ndim=1,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t F_dim_0 = F.shape[0]
    cdef np.uint16_t F_dim_1 = F.shape[1]
    cdef np.uint16_t F_dim_2 = F.shape[2]
    cdef np.uint16_t LF_dim_1 = LF.shape[1]
    cdef np.uint16_t num_LFs = LF.shape[0]
    cdef np.uint16_t num_detections = F_dim_0 - LF_dim_1 +1
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] detect_scores = np.zeros(F_dim_0,
                                                                   dtype=DTYPE)
    cdef np.ndarray[ndim=1,dtype=np.uint16_t] max_detector = np.zeros(F_dim_0,
                                                                   dtype=np.uint16)
    cdef DTYPE_t cur_detection_val
    cdef np.uint16_t cur_detector
    cdef np.uint32_t cur_detection,filter_timepoint,frequency,part_identity,lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx,x0,x1
    # cur_detection is the time point in the overall vector
    for cur_detection in range(num_detections):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
        for cur_detector in range(num_LFs):
            cur_detection_val = cs[cur_detector]
            for filter_timepoint in range(LF_dim_1):
                for frequency in range(F_dim_1):
                    for part_identity in range(F_dim_2):
                        cur_detection_val += (
                        
                            LF[cur_detector,filter_timepoint,
                               frequency,
                               part_identity] 
                            * F[cur_detection+filter_timepoint,
                            frequency,
                            part_identity])

            if cur_detector == 0:
                detect_scores[cur_detection] = cur_detection_val
                max_detector[cur_detection] = cur_detector
            else:
                if cur_detection_val > detect_scores[cur_detection]:
                    detect_scores[cur_detection] = cur_detection_val
                    max_detector[cur_detection] = cur_detector
    return detect_scores, max_detector


def detect_float2d(np.ndarray[ndim=2,
                      dtype=DTYPE_t] F,
           np.ndarray[ndim=2,
                      dtype=DTYPE_t] LF):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=3,dtype=UINT_t]
        Features for an example utterance that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    Output:
    =======
    detect_scores: np.ndarray[ndim=1,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t F_dim_0 = F.shape[0]
    cdef np.uint16_t F_dim_1 = F.shape[1]
    cdef np.uint16_t LF_dim_0 = LF.shape[0]
    cdef np.uint16_t num_detections = F_dim_0 - LF_dim_0 +1
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] detect_scores = np.zeros(F_dim_0,
                                                                   dtype=DTYPE)
    cdef np.uint32_t cur_detection,filter_timepoint,frequency,part_identity,lo_spread_0_idx,hi_spread_0_idx,lo_spread_1_idx,hi_spread_1_idx,x0,x1
    # cur_detection is the time point in the overall vector
    for cur_detection in range(num_detections):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
        for filter_timepoint in range(LF_dim_0):
            for frequency in range(F_dim_1):
                detect_scores[cur_detection] += (
                        
                        LF[filter_timepoint,
                           frequency] 
                        * F[cur_detection+filter_timepoint,
                            frequency])
    return detect_scores



def detect_spectral(np.ndarray[ndim=2,
                      dtype=DTYPE_t] F,
           np.ndarray[ndim=2,
                      dtype=DTYPE_t] LF):
    """
    Parameters:
    ===========
    F: np.ndarray[ndim=3,dtype=UINT_t]
        Features for an example utterance that we have fit with the
        intermediate features plus downsampled. Dimensions
        of F and LF are assumed to be (time,frequency,patch_type)
    LF: np.ndarray[ndim=3,dtype=DTYPE_t]
        Linear filter for computing the likelihood
    Output:
    =======
    detect_scores: np.ndarray[ndim=1,dtype=DTYPE_t]
        Performs spreading 
    """
    cdef np.uint16_t F_dim_0 = F.shape[0]
    cdef np.uint16_t F_dim_1 = F.shape[1]
    cdef np.uint16_t LF_dim_0 = LF.shape[0]
    cdef np.uint16_t num_detections = F_dim_0 - LF_dim_0 +1
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] detect_scores = np.zeros(F_dim_0,
                                                                   dtype=DTYPE)
    cdef np.uint32_t cur_detection,filter_timepoint,frequency
    # cur_detection is the time point in the overall vector
    for cur_detection in range(num_detections):
        # filter_timepoint is where we are in the filter for
        # computing these parallel convolutions
        for filter_timepoint in range(LF_dim_0):
            for frequency in range(F_dim_1):
                
                detect_scores[cur_detection] += (
                        F[cur_detection+filter_timepoint,
                          frequency]
                        * LF[filter_timepoint,
                             frequency])
    return detect_scores
