import numpy as np
from scipy import linalg
from scipy.fftpack import fft



def labels_for_edgemaps(T, sr, label_times, labels, 
                        window_length, hop_length, 
                        kernel_length):
                        
    """Returns the the labels for the edge map computation on
    a digital signal. The labels are given as start times

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

def _get_start_samples(T,window_length,hop_length,kernel_length)
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
