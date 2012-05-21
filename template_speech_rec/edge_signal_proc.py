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

def has_pattern(pattern,labels):
    pattern_length = pattern.shape[0]
    for l in xrange(labels.shape[0]-pattern_length):
        if np.all(labels[l:l+pattern_length] == pattern):
            return True
    return False



            
class Pattern_Examples:
    def __init__(self,data_files_iter,pattern,
                 sample_rate,num_window_samples,
                 num_window_step_samples,fft_length,
                 freq_cutoff,kernel_length):
        self.examples = []
        self.data_files_iter = data_files_iter
        self.pattern = pattern
        self.sample_rate = sample_rate
        self.num_window_samples = num_window_samples
        self.num_window_step_samples = num_window_step_samples
        self.fft_length = fft_length
        self.freq_cutoff = freq_cutoff
        self.kernel_length = kernel_length
        
    def __iter__(self):
        return self

    def next(self):
        while True:
            get_s,labels,label_times = self.data_files_iter.next()
            if not(has_pattern(self.pattern,labels)):
                continue
            else:
                s = get_s()
            self.labels = labels
            feature_start, \
                feature_step, num_features =\
                _get_feature_label_times(s,
                                         self.num_window_samples,
                                         self.num_window_step_samples)
            feature_labels, \
                feature_label_transitions \
                = _get_labels(label_times,
                              labels,
                              feature_start, feature_step, num_features,
                            self.sample_rate)
            self.times = get_pattern_times(self.pattern,
                                              labels,
                                              feature_label_transitions)
            self.feature_labels = feature_labels
            self.feature_label_transitions = feature_label_transitions
            # check that pattern is in the example signal
            # before doing signal processing
            if self.times:
                self.E = get_edgemap_features(s,self.sample_rate,
                                              self.num_window_samples,
                                              self.num_window_step_samples,
                                              self.fft_length,
                                              self.freq_cutoff,
                                              self.kernel_length)
                self.examples.extend([self.E[:,p[0]:p[1]] 
                                         for p in self.times])
                print "pattern_examples now has length",len(self.examples)                
            break
            

def get_pattern_examples(data_files_iter,pattern,
                         sample_rate,num_window_samples,
                         num_window_step_samples,fft_length,
                         freq_cutoff,kernel_length):
    pattern_examples = []
    while True:
        try:
            get_s,labels,label_times = data_files_iter.next()
            if not(has_pattern(pattern,labels)):
                continue
            else:
                s = get_s()                     
            feature_start, \
                feature_step, num_features =\
                _get_feature_label_times(s,
                                         num_window_samples,
                                         num_window_step_samples)
            feature_labels, \
                feature_label_transitions \
                = _get_labels(label_times,
                              labels,
                              feature_start,
                              feature_step,
                              num_features,
                              sample_rate)
            pattern_times = get_pattern_times(pattern,
                                              labels,
                                              feature_label_transitions)
            # check that pattern is in the example signal
            # before doing signal processing
            if pattern_times:
                E = get_edgemap_features(s,sample_rate,
                                         num_window_samples,
                                         num_window_step_samples,
                                         fft_length,
                                         freq_cutoff,
                                         kernel_length)
                pattern_examples.extend([E[:,p[0]:p[1]] 
                                         for p in pattern_times])
                print "pattern_examples now has length",len(pattern_examples)                
        except: # exhausted iterator
            return pattern_examples

def get_pattern_times(pattern,labels,feature_label_transitions):
    """
    Parameters
    ----------
    pattern:
       array of strings that represents the pattern we are
       looking for
    labels:
        array of strings that contains the labels for the sequence
    feature_label_transitions:
        array of positive integers that says at which edge map
        feature the next label starts
       
    Output
    ------
    pattern_times:
       tuples where the zeroth entry is the start frame of the pattern and the 1th entry
       is the next frame after the last frame of the pattern
    """
    pattern_length = pattern.shape[0]
    pattern_times = []
    for l in xrange(labels.shape[0]-pattern_length):
        if np.all(labels[l:l+pattern_length] == pattern):
            pattern_times.append((feature_label_transitions[l],
                                  feature_label_transitions[l+pattern_length]))
    return pattern_times

def get_pattern_part_times(pattern,labels,feature_label_transitions):
    """
    Parameters
    ----------
    pattern:
       array of strings that represents the pattern we are
       looking for
    labels:
        array of strings that contains the labels for the sequence
    feature_label_transitions:
        array of positive integers that says at which edge map
        feature the next label starts
       
    Output
    ------
    pattern_times:
       tuples where the zeroth entry is the start frame of the pattern and the 1th entry
       is the next frame after the last frame of the pattern
    """
    pattern_length = pattern.shape[0]
    pattern_part_times = []
    for l in xrange(labels.shape[0]-pattern_length):
        if np.all(labels[l:l+pattern_length] == pattern):
            part_times = []
            for pattern_part_id in xrange(pattern_length):
                part_times.append((feature_label_transitions[l+pattern_part_id],
                                   feature_label_transitions[l+pattern_part_id+1]))
            pattern_part_times.append(part_times)
    return pattern_part_times


def get_pattern_negative(pattern,labels,feature_label_transitions,length):
    """
    Parameters
    ----------
    pattern:
       array of strings that represents the pattern we are
       looking for
    labels:
        array of strings that contains the labels for the sequence
    feature_label_transitions:
        array of positive integers that says at which edge map
        feature the next label starts
       
    Output
    ------
    pattern_times:
       tuples where the zeroth entry is the start frame of the pattern and the 1th entry
       is the next frame after the last frame of the pattern
    """
    pattern_length = pattern.shape[0]
    pattern_times = []
    for l in xrange(labels.shape[0]-pattern_length):
        if np.all(labels[l:l+pattern_length] == pattern):
            pattern_times.append((feature_label_transitions[l],
                                  feature_label_transitions[l+pattern_length]))
    negative_pattern_time = np.random.randint()
    return pattern_times


def _get_feature_label_times(s,
                            num_window_samples,
                            num_window_step_samples):
    feature_start, feature_step, num_features = _get_windows_sample_stats(s,num_window_samples,num_window_step_samples)
    return _get_edge_sample_stats(feature_start,feature_step,num_features)

def _get_spectrogram_label_times(s,
                            num_window_samples,
                            num_window_step_samples):
    return _get_windows_sample_stats(s,num_window_samples,num_window_step_samples)

def get_spectrogram_features(s,sample_rate,num_window_samples,
                          num_window_step_samples,fft_length,
                             freq_cutoff,kernel_length,
                             preemph=.95, quantile_level=.25):
    s = _preemphasis(s,preemph)
    S = _spectrograms(s,num_window_samples, 
                      num_window_step_samples,
                      fft_length,
                      sample_rate)
    freq_idx = int(freq_cutoff/(float(sample_rate)/fft_length))
    S = S[:,:freq_idx]
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
    return S_subsampled


def get_edgemap_features(s,sample_rate,num_window_samples,
                          num_window_step_samples,fft_length,
                          freq_cutoff,kernel_length,
                           preemph=.95, quantile_level=.25):
    s = _preemphasis(s,preemph)
    S = _spectrograms(s,num_window_samples, 
                      num_window_step_samples,
                      fft_length,
                      sample_rate)
    print "Frequency cutoff was", freq_cutoff
    freq_idx = int(freq_cutoff/(float(sample_rate)/fft_length))
    S = S[:,:freq_idx]
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
    E = _edge_map(S_subsampled,quantile_level)
    return E

def get_log_spectrogram(s,sample_rate,
                        num_window_samples,
                        num_window_step_samples,
                        fft_length,
                        preemph=.95,
                        return_freqs=False):
    s = _preemphasis(s,preemph)
    S = _spectrograms(s,num_window_samples, 
                      num_window_step_samples,
                      fft_length,
                      sample_rate)
    freq_idx = np.arange(S.shape[1]) * (float(sample_rate)/fft_length)
    #S = S[:,:freq_idx]
    # correct for the shape
    # we want each row of S to correspond to a frequency
    # and we want the bottom row to represent the lowest
    # frequency
    if return_freqs:
        return np.log(S.transpose()), freq_idx
    else:
        return np.log(S.transpose())

def get_mel_spec(s,sample_rate,
                        num_window_samples,
                        num_window_step_samples,
                        fft_length,
                        preemph=.95,
                        freq_cutoff=None,
                 nbands = 40):
    """
    The mel spectrogram, this is the basis for MFCCs
    """
    S,freq_idx = get_log_spectrogram(s,sample_rate,
                               num_window_samples,
                               num_window_step_samples,
                               fft_length,
                               preemph=.95,
                               return_freqs=True)
    return audspec(S,sample_rate,nbands=nbands,
                max_freq=freq_cutoff)

def get_edgemap_no_threshold(s,sample_rate,
                             num_window_samples,
                             num_window_step_samples,
                             fft_length,
                             freq_cutoff,kernel_length,
                             preemph=.95):
    s = _preemphasis(s,preemph)
    S = _spectrograms(s,num_window_samples, 
                      num_window_step_samples,
                      fft_length,
                      sample_rate)
    freq_idx = int(freq_cutoff/(float(sample_rate)/fft_length))
    S = S[:,:freq_idx]
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
    return _edge_map_no_threshold(S_subsampled)

def threshold_edgemap(E,quantile_level,
                      edge_feature_row_breaks,
                      report_level=False,
                      abst_threshold=-np.inf*np.ones(8)):
    # see whether to report the level of the edge thresholds
    if report_level:
        edge_thresholds = np.empty(8)
    # allocate an empty array for the thresholded edges
    E_new = np.empty(E.shape)
    for edge_feat_idx in xrange(1,edge_feature_row_breaks.shape[0]):
        start_idx = edge_feature_row_breaks[edge_feat_idx-1]
        end_idx = edge_feature_row_breaks[edge_feat_idx]
        if report_level:
            E_new[start_idx:end_idx,:],edge_thresholds[edge_feat_idx-1] = \
                threshold_edge_block(E[start_idx:end_idx,:],quantile_level,report_level,
                                     abst_threshold[edge_feat_idx-1])
        else:
            E_new[start_idx:end_idx,:] = \
                threshold_edge_block(E[start_idx:end_idx,:],
                                     quantile_level,
                                     report_level,
                                     abst_threshold[edge_feat_idx-1])
    if report_level:
        return E_new, edge_thresholds



def threshold_edge_block(E_block,quantile_level,
                         report_level,
                         abst_threshold):
    maxima_idx = E_block > -np.inf
    maxima_vals = E_block[maxima_idx].ravel().copy()
    maxima_vals.sort()
    tau_quant = maxima_vals[int(quantile_level*maxima_vals.shape[0])].copy()
    # zero out everything less than the quantile
    A = E_block[maxima_idx]
    # get the indices for the significant edges
    sig_idx = E_block[maxima_idx] > max(tau_quant,abst_threshold)
    A[np.logical_not(sig_idx)] = 0.
    A[sig_idx] =1
    E_block[maxima_idx] = A
    E_block[np.logical_not(maxima_idx)]=0
    if report_level:
        return E_block,tau_quant
    else:
        return E_block


def _compute_max_edges(Cand_max,Cmp1,Cmp2):
    """ Compute maximal edges and set the non-maximal edges
    to -inf
    """
    non_maxima_idx = np.logical_or(np.logical_or(Cand_max<
                               Cmp1,
                               Cand_max<
                               Cmp2), 
                                   np.logical_and(
            Cand_max == Cmp1,
            Cand_max == Cmp2))
    Cand_max[non_maxima_idx]=-np.inf
    return Cand_max


def _compute_max_and_threshold(Cand_max,Cmp1,Cmp2,
                               quantile_level):
    non_maxima_idx = np.logical_or(np.logical_or(Cand_max<
                               Cmp1,
                               Cand_max<
                               Cmp2), 
                                   np.logical_and(
            Cand_max == Cmp1,
            Cand_max == Cmp2))
    Cand_max[non_maxima_idx]=0.
    maxima_idx = np.logical_not(non_maxima_idx)
    # perform thresholding
    maxima_vals = Cand_max[maxima_idx].flat.copy()
    maxima_vals.sort()
    tau_quant = maxima_vals[int(quantile_level*maxima_vals.shape[0])].copy()
    # zero out everything less than the quantile
    A = Cand_max[maxima_idx]
    # get the indices for the significant edges
    sig_idx = Cand_max[maxima_idx] < tau_quant
    A[np.logical_not(sig_idx)] = 0.
    A[sig_idx] =1
    Cand_max[maxima_idx] = A
    return Cand_max

def spread_edgemap(T,edge_feature_row_breaks,edge_orientation,spread_length=3):
    for break_idx in xrange(1,edge_feature_row_breaks.shape[0]):
        start_idx = edge_feature_row_breaks[break_idx-1]
        end_idx = edge_feature_row_breaks[break_idx]
        T[start_idx:end_idx,:] = \
            _spread_edges(T[start_idx:end_idx],
                          edge_orientation[break_idx-1,:],
                          spread_length)

def _spread_edges(T,direction,spread_length):
    Z = np.zeros(T.shape+np.array((2*spread_length,2*spread_length)))
    Z[spread_length:-spread_length,spread_length:-spread_length] = T.copy()
    T_spread = np.zeros(T.shape)
    F_len = T.shape[0]
    T_len = T.shape[1]
    for k in xrange(-spread_length,spread_length+1):
        T_spread = np.maximum(T_spread,Z[spread_length+k*direction[0]:
                                             spread_length+F_len+k*direction[0],
                                         spread_length+k*direction[1]:
                                             spread_length+T_len+k*direction[1]])
    return T_spread


def _edge_map_no_threshold(S):
    """ function to do the edge processing
    somewhat complicated we have eight different directions
    for the edges to run
    consider the direction [1,0]

    indices range over [0,...,F-1],[0,...,T-1]
    in this case the entry
    E[0,0] = 0 if S[2,0] - S[1,0] < max(S[1,0]-S[0,0],
                                        S[3,0]-S[2,0])
            
    E[i,j] = 0 if S[i+2,j]-S[i+1,j] < max(S[i+1,j]-S[i,j],
d                                          S[i+2,j]-S[i+1,j])

    T_diff = T[1:,:]-T[:-1,:]
    T_bigger_left = (T_diff[1:-1,:]>T_diff[:-2,:])
    T_bigger_right = (T_diff[1:-1,:]>T_diff[2:,:])
    
    T_other_diff = -T_diff
    T_other_big_left = (T_other_diff[1:-1,:]-T_other_diff[:-2,:]) > 0.
    T_other
    S[2:-1,:] - S[1:-2,:]
    in the case [-1,0]
    E[0,0]=0 if S[1,0]-S[2,0] < max(S[0,0]-S[1,0],
                                    S[2,0]-S[3,0])
    
    E[i,j] = 0 if S[i+1,j]-S[i+2,j] < max(S[i,j]-S[i+1,j],
                                          S[i+1,j]-S[i+2,j])
                                    
    [0,1]
    E[i,j]
    """
    edge_feature_row_breaks = np.zeros(9)
    edge_orientations = np.zeros((8,2))
    # get [1,0] and [-1,0] features
    cur_E_idx = 0
    E = np.empty((8*S.shape[0]-6*3,S.shape[1]-3))
    T = S.copy()
    # cut down time
    T_diff = T[1:,1:-2] - T[:-1,1:-2]
    T_use = _compute_max_edges(T_diff[1:-1,:],
                         T_diff[:-2,:],
                         T_diff[2:,:])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[0,0]=1
    edge_orientations[0,1]=0
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = -T[1:,1:-2] + T[:-1,1:-2]
    T_use = _compute_max_edges(T_diff[1:-1,:],
                         T_diff[:-2,:],
                         T_diff[2:,:])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    # keep track of where these edge features started
    edge_orientations[1,0]=-1
    edge_orientations[1,1]=0
    edge_feature_row_breaks[1] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]    
    # edge is [0,1]
    T_diff = T[:,1:]- T[:,:-1]
    T_use = _compute_max_edges(T_diff[:,1:-1],
                         T_diff[:,:-2],
                         T_diff[:,2:])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[2,0]=0
    edge_orientations[2,1]=1
    edge_feature_row_breaks[2] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = - T[:,1:] + T[:,:-1]
    T_use = _compute_max_edges(T_diff[:,1:-1],
                                       T_diff[:,:-2],
                                       T_diff[:,2:])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[3,0]=0
    edge_orientations[3,1]=-1
    edge_feature_row_breaks[3] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]    
    # edge is [1,1]
    T_diff = T[1:,1:] - T[:-1,:-1]
    T_use = _compute_max_edges(T_diff[1:-1,1:-1],
                                       T_diff[:-2,:-2],
                                       T_diff[2:,2:])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[4,0]=1
    edge_orientations[4,1]=1
    edge_feature_row_breaks[4] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = - T[1:,1:] + T[:-1,:-1]
    T_use = _compute_max_edges(T_diff[1:-1,1:-1],
                               T_diff[:-2,:-2],
                               T_diff[2:,2:])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[5,0]=-1
    edge_orientations[5,1]=-1
    edge_feature_row_breaks[5] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]
    # edge [-1,1]
    T_diff = T[1:,:-1] - T[:-1,1:]
    T_use = _compute_max_edges(T_diff[1:-1,1:-1],
                               T_diff[:-2,2:],
                               T_diff[2:,:-2])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[6,0]=1
    edge_orientations[6,1]=-1
    edge_feature_row_breaks[6] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = - T[1:,:-1] + T[:-1,1:]
    T_use = _compute_max_edges(T_diff[1:-1,1:-1],
                                       T_diff[:-2,2:],
                                       T_diff[2:,:-2])
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_use.copy()
    edge_orientations[7,0]=-1
    edge_orientations[7,1]=1
    edge_feature_row_breaks[7] = cur_E_idx
    cur_E_idx = cur_E_idx + T_use.shape[0]
    edge_feature_row_breaks[8] = cur_E_idx
    return E,edge_feature_row_breaks, edge_orientations


def _edge_map(S,quantile_level,spread_length=3):
    """ function to do the edge processing
    somewhat complicated we have eight different directions
    for the edges to run
    consider the direction [1,0]

    indices range over [0,...,F-1],[0,...,T-1]
    in this case the entry
    E[0,0] = 0 if S[2,0] - S[1,0] < max(S[1,0]-S[0,0],
                                        S[3,0]-S[2,0])
            
    E[i,j] = 0 if S[i+2,j]-S[i+1,j] < max(S[i+1,j]-S[i,j],
d                                          S[i+2,j]-S[i+1,j])

    T_diff = T[1:,:]-T[:-1,:]
    T_bigger_left = (T_diff[1:-1,:]>T_diff[:-2,:])
    T_bigger_right = (T_diff[1:-1,:]>T_diff[2:,:])
    
    T_other_diff = -T_diff
    T_other_big_left = (T_other_diff[1:-1,:]-T_other_diff[:-2,:]) > 0.
    T_other
    S[2:-1,:] - S[1:-2,:]
    in the case [-1,0]
    E[0,0]=0 if S[1,0]-S[2,0] < max(S[0,0]-S[1,0],
                                    S[2,0]-S[3,0])
    
    E[i,j] = 0 if S[i+1,j]-S[i+2,j] < max(S[i,j]-S[i+1,j],
                                          S[i+1,j]-S[i+2,j])
                                    
    [0,1]
    E[i,j]
    """
    # get [1,0] and [-1,0] features
    cur_E_idx = 0
    E = np.empty((8*S.shape[0]-6*3,S.shape[1]-3))
    T = S.copy()
    # cut down time
    T_diff = T[1:,1:-2] - T[:-1,1:-2]
    T_use = _compute_max_and_threshold(T_diff[1:-1,:],
                                       T_diff[:-2,:],
                                       T_diff[2:,:],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(1,0))
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = -T[1:,1:-2] + T[:-1,1:-2]
    T_use = _compute_max_and_threshold(T_diff[1:-1,:],
                                       T_diff[:-2,:],
                                       T_diff[2:,:],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(1,0))
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]    
    # edge is [0,1]
    T_diff = T[:,1:]- T[:,:-1]
    T_use = _compute_max_and_threshold(T_diff[:,1:-1],
                                       T_diff[:,:-2],
                                       T_diff[:,2:],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(0,1))
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = - T[:,1:] + T[:,:-1]
    T_use = _compute_max_and_threshold(T_diff[:,1:-1],
                                       T_diff[:,:-2],
                                       T_diff[:,2:],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(0,1))
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]    
    # edge is [1,1]
    T_diff = T[1:,1:] - T[:-1,:-1]
    T_use = _compute_max_and_threshold(T_diff[1:-1,1:-1],
                                       T_diff[:-2,:-2],
                                       T_diff[2:,2:],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(1,1)) 
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = - T[1:,1:] + T[:-1,:-1]
    T_use = _compute_max_and_threshold(T_diff[1:-1,1:-1],
                                       T_diff[:-2,:-2],
                                       T_diff[2:,2:],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(1,1)) 
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]
    # edge [-1,1]
    T_diff = T[1:,:-1] - T[:-1,1:]
    T_use = _compute_max_and_threshold(T_diff[1:-1,1:-1],
                                       T_diff[:-2,2:],
                                       T_diff[2:,:-2],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(1,-1))
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]
    T_diff = - T[1:,:-1] + T[:-1,1:]
    T_use = _compute_max_and_threshold(T_diff[1:-1,1:-1],
                                       T_diff[:-2,2:],
                                       T_diff[2:,:-2],
                                       quantile_level)
    T_spread = _spread_edges(T_use,(1,-1))     
    E[cur_E_idx:cur_E_idx+T_use.shape[0],:] = T_spread.copy()
    cur_E_idx = cur_E_idx + T_use.shape[0]
    return E

    


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

def _get_windows_sample_stats(s,num_window_samples,num_window_step_samples):
    # compute the average sample index for each window
    # this allows us to associate a time to each entry of the spectrogram
    num_windows = (len(s)-num_window_samples)/num_window_step_samples+1
    first_window_s_avg = (num_window_samples-1)/2.
    window_s_avg_step = num_window_step_samples
    return first_window_s_avg, window_s_avg_step, num_windows

def _get_smoothed_sample_avg(window_avgs,kernel_length):
    start_smoothed = (window_avgs[0] + window_avgs[kernel_length-1])/2.
    end_smoothed = (window_avgs[-1]+window_avgs[-kernel_length])/2.
    win_diffs = window_avgs[1] - window_avgs[0]
    return np.arange(start_smoothed,end_smoothed+.1,win_diffs)

def _get_edge_sample_stats(feature_start,feature_step, num_features):
    # we ignore the first and last because we only look
    # at maxima for edges that stretch along time
    # we also consider the time in between two edges as the location of the edge
    # 3/2, 5/2, ... ,num_features-5/2
    # 0, 1, ... , num_edge_features-1
    # num_edge_features  = num_features - 5/2 -3/2 +1
    return feature_start + 3./2 * feature_step,\
        feature_step,\
        num_features - 3
        

def _get_labels(label_times,
                labels,
                feature_start, feature_step, num_features,
                sr):
    # label_start_times and labels should have the same length feature
    # samples is an array of what sample each feature in a spectrogram
    # or edgegram corresponds to
    # label_start_times are the absolute times when features start
    # labels are the labels themselves, we will construct an array the same size as feature_samples but whose entries are the label_start_times
    # sr is the sampling rate
    label_transitions = sr * label_times
    feature_labels = np.empty(num_features,
                              dtype=labels.dtype)
    end_idx = -1
    # make sure that we start where the label_transition
    # surrounds the feature starting index
    label_time_start_idx = 0
    while label_transitions[label_time_start_idx,1] <= feature_start:
        label_time_start_idx +=1
    # initialize array that keeps track of when transitions
    # in the labels happen
    feature_label_transitions = -np.int_(np.ones(label_times.shape[0]))
    for ls_time_idx in xrange(label_time_start_idx,label_times.shape[0]):
        # find smallest k such that
        #    label_transitions[ls_time_idx,0] <= feature_start
        #          + k * feature_step
        start_idx = end_idx+1
        # find largest k such that
        #    label_transitions[ls_time_idx,1] > feature_start
        #          + k * feature_step
        end_idx = min(int(np.floor((label_transitions[ls_time_idx,1] -\
            feature_start)/feature_step)),num_features-1)
        feature_labels[start_idx:end_idx+1] =labels[ls_time_idx]
        feature_label_transitions[ls_time_idx] = np.int(start_idx)
    return feature_labels, feature_label_transitions
    

def _spectrograms(s,num_window_samples, 
                  num_window_step_samples,
                  fft_length,
                  sample_rate):
    # pre-emphasis 
    s=_preemphasis(s)
    windows = _get_windows(s,num_window_samples,num_window_step_samples)
    swindows = np.vectorize(lambda i: s[i])(windows)
    return np.abs(fft(hanning(num_window_samples) * swindows,fft_length)[:,:fft_length/2+1])


def audspec(spectrogram,sample_rate,nbands=None,
            min_freq=0,max_freq=None,
            sumpower=1,bwidth=1.0):
    """
    Copied from http://labrosa.ee.columbia.edu/matlab/rastamat/audspec.m
    sample_rate should be an integer
    """
    if nbands is None:
        nbands = int(np.ceil(hz2bark(sample_rate/2)+1))
    if max_freq is None:
        max_freq = sample_rate/2
    nfreqs, nframes = spectrogram.shape
    nfft = (nfreqs-1)*2
    # only implementing the mel case
    wts,_ = fft2melmx(nfft,sample_rate,nbands,bwdith,
                    minfreq,maxfreq)
    wts = wts[:, :nfreqs]
    return np.dot(wts, spectrogram)
    

def fft2melmx(nfft,sample_rate,nfilts=40,width=1.0,minfrq=0,
              maxfrq=None):
    """
    Copied nearly directly from Dan Ellis' code:
    http://labrosa.ee.columbia.edu/matlab/rastamat/fft2melmx.m
    
    Parameters:
    ===========
    nfft: int
        number of samples for the fourier transform
    sr: int
        number of samples per second

    Complete, matches with Dan Ellis' implementation
    """
    if maxfrq is None:
        maxfrq = sample_rate/2
    wts = np.zeros((nfilts,nfft))
    # center frequencies for the mel bins
    fftfrqs = np.arange(nfft)/np.float64(nfft)*sample_rate
    minmel = hz2mel_num(minfrq)
    maxmel = hz2mel_num(maxfrq)
    binmels = np.arange(minmel,maxmel+1,(maxmel-minmel)/(nfilts+1))
    binfrqs = np.array(map(mel2hz,
                           binmels))
    binbin = np.array(map(round,binfrqs/sample_rate*(nfft-1)))
    for filt in xrange(nfilts):
        fs = binfrqs[filt:filt+3].copy()
        fs = fs[1] + width * (fs - fs[1])
        loslope = (fftfrqs - fs[0])/(fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs)/(fs[2] - fs[1])
        wts[filt,:] = np.maximum(0,
                              np.minimum(loslope, 
                                         hislope))
    wts = np.dot(np.diag(2./(binfrqs[2+np.arange(nfilts)]-binfrqs[:nfilts])),wts)
    wts[:,(nfft/2+1):nfft+1]=0.
    return wts, binfrqs

    

def mel2hz(mel):
    f_0 = 0; # 133.33333;
    f_sp = 200./3; # 66.66667;
    brkfrq = 1000;
    # starting mel value for log region
    brkpt  = (brkfrq - f_0)/f_sp;  
    if mel > brkpt:        
        # log(exp(log(6.4)/27))
        # magic log step number
        logstep = 0.068751777420949123
        return brkfrq*np.exp(logstep*(mel-brkpt))
    else:
        return f_0 + f_sp*mel

      

def hz2mel_num(freq):
    """
    Copied from Dan Ellis' code
    http://labrosa.ee.columbia.edu/matlab/rastamat/hz2mel.m
    
    Completed
    """
    f_0 = 0.
    f_sp = 200/3.
    brkfrq = 1000.
    if  freq < brkfrq:
        return (freq - f_0)/f_sp
    else:
        brkpt = (brkfrq - f_0)/f_sp
        step = 0.068751777420949123 # np.log(6.4)/27
        return brkpt + np.log(freq/brkfrq)/step
    

def htk_hz2mel(freq):
    """
    Copied from Dan Ellis' code
    http://labrosa.ee.columbia.edu/matlab/rastamat/hz2mel.m
    Based on the mel frequency bin computations from the htk kit
    """
    return 2595. * np.log10(1+freq/700.)

def hz2bark(freq):
    return 6. * np.arcsinh(freq/600.)

def _mel_filter_freq():    
    
    pass
 

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
