#
# Author: Mark Stoehr
#
#


##
#
# Generally handles the audio processing and labeling
#
#

import numpy as np
from scipy import linalg
from scipy.fftpack import fft
from scipy.signal.windows import hanning


def do_feature_processing(s,spectrogram_parameters,
                          kernel_parameters,
                          preemphasis=.95):
    
    pass

def _preemphasis(s):
    return np.concatenate([s[0],
                           s[1:]- preemphasis*s[:-1]]) 

def _preemphasis_times(s,sr):
    return 1./sr * np.arange(len(s));

def _get_windows(s,num_window_samples,num_window_step_samples):
    # we have the property that we have windows with indices
    # 0 ... last_win_idx
    # the sample index for the first entry of window last_win_idx
    # will be last_win_idx*num_window_step_samples
    # so its the largest integer such that 
    # last_win_idx*num_window_step_samples + num_window_samples -1 <= len(s)-1

    # division computes floor implicitly
    last_win_idx = (len(s)-num_window_samples)/num_window_step_samples
    num_windows = last_win_idx + 1
    window_idx = np.arange(num_window_samples)
    return tiles(np.arange(num_window_samples),
                    (num_windws,1))\
                    + num_window_step_samples\
                    *np.arange(num_windows).reshape((num_windows,1))

def _get_windows_sample_avg(s,num_window_samples,num_window_step_samples):
    # compute the average sample index for each window
    # this allows us to associate a time to each entry of the spectrogram
    num_windows = (len(s)-num_window_samples)/num_window_step_samples+1
    window_avgs = np.arange(num_windows) + (num_window_samples-1)/2.
    return window_avgs

def _get_smoothed_sample_avg(window_avgs,kernel_length):
    start_smoothed = (window_avgs[0] + window_avgs[kernel_length-1])/2.
    end_smoothed = (window_avgs[-1]+window_avgs[-kernel_length])/2.
    win_diffs = window_avgs[1] - window_avgs[0]
    return np.arange(start_smoothed,end_smoothed+.1,win_diffs)

def _get_edge_sample_avg(smoothed_avgs):
    # we ignore the first and last because we only look
    # at maxima for edges that stretch along time
    # for purely vertical edges its merely the second start time
    # need to remove two of the vertical edge features
    start_sample = (smoothed_avgs[1]+3*(smoothed_avgs[2]+smoothed_avgs[1])/2.)/4.
    end_sample = (smoothed_avgs[-3]+3*(smoothed_avgs[-2]+smoothed_avgs[-3])/2.)/4.
    sample_diffs = smoothed_avgs[1]-smoothed_avgs[0]
    return np.arange(start_sample, end_sample+.1,smoothed_diffs)

def _get_labels(label_start_times,labels,
                feature_start_sample,
                feature_end_sample,
                feature_diff,sr):
    # label_start_times and labels should have the same length feature
    # samples is an array of what sample each feature in a spectrogram
    # or edgegram corresponds to
    # label_start_times are the absolute times when features start
    # labels are the labels themselves, we will construct an array the same size as feature_samples but whose entries are the label_start_times
    # sr is the sampling rate
    feature_transitions = int(sr * label_times)
    feature_labels = np.empty(feature_samples.shape,
                              dtype=labels.dtype)
    
    cur_feature_idx = 0
    # skip the first index becase we are intrerest in the intervals
    for ls_time_idx in xrange(1,label_start_times.shape[0]):
        # convert times to indices
        # feature_start_sample + k*feature_diff < sr*label_start_times[ls_time_idx]
        next_feature_idx = int(ceil((sr*label_start_times[ls_time_idx]-feature_start_sample)/feature_diff))
        feature_labels[cur_feature_idx:]
        
    
    pass

def _spectrograms(s,num_window_samples, 
                  num_window_step_samples,
                  fft_length,freq_cutoff,
                  sample_rate):
    # pre-emphasis 
    s=_preemphasis(s)
    windows = _get_windows(s,num_window_samples,num_window_step_samples)
    swindows = np.vectorize(lambda i: s[i])(windows)
    freq_idx = freq_cutoff/(float(sample_rate)/num_window_samples)
    return np.abs(fft(hanning(num_window_samples) * swindows,)[:,:freq_idx])
    
    # 
    

def edge_map_times(N, window_length, hop_length,
                   kernel_length):
    """Returns a vector of times, each entry corresponds to
    frame in the edge map features, the entry value is the
    time in the signal that these features are considered
    to occur.  This is used in the process of labeling
    the phonetic class that particular edge map features
    are associated with 

    The assumption is that the signal is uniformly sampled. 
    The times returned will sometimes occur within a sample

    Parameters
    ----------
    N: int
       Number of samples in the signal
       
    window_length:
       Number of samples in a given window

    hop_length:
       Number of samples we jump over in a hop

    kernel_length:
       Number of samples that we take the kernel over

    """
    # compute the number of windows
    num_windows = (N-window_length)/hop_length+1
    window_points = window_length/2. \
        + hop_length * np.arange(num_windows)
    
    # correct for kernel length
    window_points = window_points[window_length/2:\
                                      end-window_length/2]

    window_points = window_points[1:end-1]
    


def labels_for_edgemaps(T, sr, label_times, labels, 
                        window_length, hop_length, 
                        kernel_length):
                        
    """Returns the the labels for the edge map computation on
    a digital signal. The labels are given as start times for
    when a given sound begins

    Parameters
    ----------
    T: int
        number of samples in signal
    sr: int
        samples per second
    label_times: array
        N-dimensional array of floats, each time should be
        less than T/sr 
    labels: ndarray
        labels can be any datatype, should be the same length
        as label_times
    window_length: int
        number of samples in a window
    hop_length: int
        number of samples between successive windows
    kernel_length: int
        number of frames used in the smoothing kernel
        
    Returns
    -------
    edge_map_labels: array
        array of the labels, the length of the array
        corresponds to the length of the edge map features
    edge_map_transitions: ndarray
        an N-dimensional array of integers where the integers
        correspond to when the next label starts and N is
        the number of labels
    """
    start_sample, num_windows = _get_start_samples(T,
                                                   window_length,
                                                   hop_length,
                                                   kernel_length)
    # correct for where maxima can occur
    start_sample += hop_length
    # maxima can only occur in the middle windows
    num_windows = num_windows - 2
    # convert to label times to sample indices
    edge_map_transitions = np.ceil(sr * label_times)
    edge_map_labels = np.empty([num_windows],dtype=labels.dtype)
    # we find all the samples that for a given t are such 
    # that they are in the half-open interval
    #  \[edge_map_transitions[t],
    #    edge_map_transitions[t+1] \)
    cur_map = 0
    for t in range(label_times.size-1):
        num_edge_maps = (edge_map_transitions[t+1] \
                       - edge_map_transitions[t] ) \
                       / hop_length
        edge_map_labels[cur_map:cur_map+num_edge_maps] \
            = labels[t]
        cur_map=cur_map+num_edge_maps
    edge_map_labels[cur_map:-1]=labels[-1]
    return edge_map_labels, edge_map_transitions

def _get_start_samples(T,window_length,hop_length,kernel_length):
    """
    Returns the start and end window sample indices for the 
    frames in the short time fourier transform after
    smoothing with a kernel over kernel_length frames

    Parameters
    ----------
    T: int
        number of samples in the signal
    window_length: int
        number of samples in a window
    hop_length: int
        number of samples between successive windows
    kernel_length: int
        number of windows used by the smoothing kernel

    Returns
    -------
    start_sample: double
        sample index for the middle sample of the first 
        window (possibly not an integer)
    num_windows: int
        total number of windows

    """
    # the kernel_length/2 +1 takes care of the boundary 
    # windows lost to smoothing
    start_window = kernel_length/2+1
    # the -2 takes care of the fact we lose one window
    #  at the end for edges across time
    end_window = (T-window_length)/hop_length \
        - kernel_length/2 -2
    num_windows = end_window - start_window + 1
    # want the exact middle of window
    window_middle = (window_length-1)/2.
    start_sample = start_window*hop_length + window_middle
    return start_sample, num_windows
