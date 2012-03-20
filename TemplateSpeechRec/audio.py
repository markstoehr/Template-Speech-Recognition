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
from scipy.signal import convolve


class SpectrogramClass:
    def __init__(self,num_window_samples,num_window_step_samples,
                 sample_rate,fft_length,freq_cutoff):
        self.num_window_samples = num_window_samples
        self.num_window_step_samples = num_window_step_samples
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.freq_cutoff = freq_cutoff



def do_feature_processing(s,sample_rate,num_window_samples,
                          num_window_step_samples,fft_length,
                          freq_cutoff,kernel_length,
                           preemph=.95):
    s_avgs = _get_windows_sample_avg(s,num_window_samples,num_window_step_samples)
    s_avgs = _get_smoothed_sample_avg(s_avgs,kernel_length)
    s_avgs = _get_edge_sample_avg(s_avgs)
    s = _preemphasis(s,preemph)
    S = _spectrograms(s,num_window_samples, 
                      num_window_step_samples,
                      fft_length,freq_cutoff,
                      sample_rate)
    # correct for the shape
    # we want each row of S to correspond to a frequency
    # and we want the bottom row to represent the lowest
    # frequency
    S = np.log(S.transpose())
    #S = S[::-1,:]
    # smooth the spectrogram
    smoothing_kernel = make_gaussian_kernel(kernel_length)
    S_smoothed = convolve(S,smoothing_kernel,mode = "same")
    S_subsampled = S_smoothed[::2,:]
    # compute the edgemap
    E = _edge_map(S_subsampled)
    return E, s_avgs

def _edge_map(S):
    """ function to do the edge processing
    somewhat complicated we have eight different directions
    for the edges to run
    consider the direction [1,0]

    indices range over [0,...,F-1],[0,...,T-1]
    in this case the entry
    E[0,0] = 0 if S[2,0] - S[1,0] < max(S[1,0]-S[0,0],
                                        S[3,0]-S[2,0])
            
    E[i,j] = 0 if S[i+2,j]-S[i+1,j] < max(S[i+1,j]-S[i,j],
                                          S[i+2,j]-S[i+1,j])

    T_diff = T[1:,:]-T[:-1,:]
    T_bigger_left = (T_diff[1:-1,:] - T_diff[:-2,:])>0.
    T_bigger_right = (T_diff[1:-1,:]-T_diff[2:,:])>0.
    
    T_other_diff = -T_diff
    T_other_big_left = (T_other_diff[1:-1,:])
    S[2:-1,:] - S[1:-2,:]
    in the case [-1,0]
    E[0,0]=0 if S[1,0]-S[2,0] < max(S[0,0]-S[1,0],
                                    S[2,0]-S[3,0])
    
    E[i,j] = 0 if S[i+1,j]-S[i+2,j] < max(S[i,j]-S[i+1,j],
                                          S[i+1,j]-S[i+2,j])
                                    
    [0,1]
    E[i,j]
    """
    edge_diffs = _get_edge_diffs(S)
    edge_maxima = _get_edge_maxima(edge_diffs)
    edge_threshold = _get_edge_threshold(edge_maxima,.7)
    
    edge_orientations = np.array([[-1,-1],
                                  [-1,0],
                                  [-1,1],
                                  [0,-1],
                                  [0,1],
                                  [1,-1],
                                  [1,0],
                                  [1,1]])


def _get_edge_diffs(S):
    # edge [0,-1]
    top_bottom = S[:,:-1] - S[:,1:]
    
    # edge [0,1]
    bottom_top = -top_bottom
    # edge [-1,0]
    left_right = S[:-1,:] - S[1:,:]
    # edge [1,0]
    right_left = -left_right
    # edge [-1,-1]
    lefttop_rightbot = S[:-1,:-1] - S[1:,1:]
    # edge [1,1]
    rightbot_lefttop = - lefttop_rightbot
    # edge [1,-1]
    righttop_leftbot = S[1:,:-1] - S[:-1,1:]
    # edge [-1,1]
    leftbot_righttop=-righttop_leftbot
    

    

def _preemphasis(s,preemph=.95):
    return np.concatenate([[s[0]],
                           s[1:]- preemph*s[:-1]]) 


def make_gaussian_kernel(size,fwhm=3):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.

    code is (slightly) modified from:
    http://mail.scipy.org/pipermail/scipy-user/2006-June/008366.html
    """
    x = np.arange(0, size, 1, np.float64)
    y = x[:,np.newaxis]
    x0 = y0 = size // 2
    g=np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return g/g.sum()


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
    return np.tile(np.arange(num_window_samples),
                    (num_windows,1))\
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

def _get_labels(label_start_times,label_end_times,
                labels,
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
    num_feature_labels = feature_labels.shape[0]
    last_time = label_end_times[-1]
    feature_end_idx = label_end_times/last_time \
        * (num_feature_labels-1)
    feature_transitions = np.concatenate([np.array([0]),feature_end_idx.astype(int)])
    for ls_time_idx in xrange(len(feature_transitions)-1):
        start_idx = feature_transitions[ls_time_idx]
        end_idx = feature_transitions[ls_time_idx+1]
        feature_labels[start_idx:end_idx] =labels[ls_time_idx]
    return feature_labels, feature_transitions
    

def _spectrograms(s,num_window_samples, 
                  num_window_step_samples,
                  fft_length,freq_cutoff,
                  sample_rate):
    # pre-emphasis 
    s=_preemphasis(s)
    windows = _get_windows(s,num_window_samples,num_window_step_samples)
    swindows = np.vectorize(lambda i: s[i])(windows)
    freq_idx = int(freq_cutoff/(float(sample_rate)/fft_length))
    return np.abs(fft(hanning(num_window_samples) * swindows,fft_length)[:,:freq_idx])


    
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
