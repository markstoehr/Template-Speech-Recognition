import numpy as np
from scipy import ndimage

import sys, os, cPickle,re,itertools,collections,operator

import bernoulli_mixture as bernoulli_mixture
import edge_signal_proc as esp
import extract_local_features as elf
import code_parts as cp
import spread_waliji_patches as swp
import compute_likelihood_linear_filter


class AverageBackground:
    def __init__(self):
        self.num_frames = 0
        self.processed_frames = False
    # Method to add frames
    def add_frames(self,E,edge_feature_row_breaks=None,
                   edge_orientations=None,abst_threshold=None,
                   time_axis=1):
        new_E = E.copy()
        if abst_threshold is not None:
            esp._edge_map_threshold_segments(new_E,
                                             40,
                                             1,
                                             threshold=.3,
                                             edge_orientations = edge_orientations,
                                             edge_feature_row_breaks = edge_feature_row_breaks)
        if not self.processed_frames:
            self.E = np.mean(new_E,axis=time_axis)
            self.processed_frames = True
        else:
            self.E = (self.E * self.num_frames + np.sum(new_E,axis=time_axis))/(self.num_frames+new_E.shape[time_axis])
        self.num_frames += new_E.shape[time_axis]


def reorg_part_for_fast_filtering(part,feature_types=8):
    """
    Assumes the patch for different edge types have been vertically stacked
    and that there are eight edge types
    dimensions are features by time
    want time by feature by edge type
    """
    H = part.shape[0]/feature_types
    return np.array([
            part[i*H:(i+1)*H].T
            for i in xrange(feature_types)]).swapaxes(0,1).swapaxes(1,2)

def reorg_parts_for_fast_filtering(parts,feature_types=8,min_prob = .01):
    filters =  np.clip(np.array([
            reorg_part_for_fast_filtering(part,feature_types=feature_types)
            for part in parts]),min_prob,1-min_prob)
    return np.log(filters).astype(np.float32), np.log(1-filters).astype(np.float32)


def get_data_files_indices(train_data_path):
    num_pattern = re.compile('[0-9]+s.npy')
    return [s_name[:-len('s.npy')] for fname in os.listdir(train_data_path)
     for s_name in num_pattern.findall(fname)]


def phns_syllable_matches(phns,syllable):
    syllable_len = len(syllable)
    return np.array([ phn_id
                      for phn_id in xrange(len(phns)-syllable_len+1)
                      if np.all(phns[phn_id:phn_id+syllable_len]==np.array(syllable))])

def collapse_to_grid(E_coded,grid_time,grid_frequency):
    """
    Parameters:
    ===========
    E_coded: numpy.ndarray[ndim=2,dtype=int]
        Feature map that indicates the presence of the waliji feature
    grid_time: int
    grid_frequency: int
    """
    return E_coded[::grid_time,::grid_frequency]

def map_array_to_coarse_coordinates(xs,
                                    coarsen_factor):
    return ((xs + coarsen_factor/2.)/coarsen_factor).astype(int)

def get_examples_from_phns_ftls(syllable,
                                phns,
                                flts,
                                log_part_blocks,
                                coarse_length):
    """
    Parameters:
    ===========
    syllables: list of np.ndarray[ndim=1,dtype=str]
        Syllable that we are looking for in the phns
    phns: np.ndarray[ndim=1, dtype=str]
        The phones that occur in the utterance
        and correspond to the time breaks in ftls
    ftls: np.ndarray[ndim=1, dtype=int]
        Feature label transitions, these are the time
        points where a new phone label begins.
    log_part_blocks: np.ndrray[ndim=4,dtype=np.float32]
        contains the model for the features, main
        data from this is knowing what the subsampling parameters
        are going to work
    """
    if log_part_blocks is None:
        coarse_factor = 1
    else:
        coarse_factor = log_part_blocks.shape[1]
    phn_matches = phns_syllable_matches(phns,syllable)
    if phn_matches.shape[0] ==0:
        return None, None
    syllable_starts = flts[ phn_matches]
    syllable_ends = flts[phn_matches + len(syllable)]
    syllable_starts = map_array_to_coarse_coordinates(syllable_starts,coarse_factor)
    syllable_ends = np.clip(map_array_to_coarse_coordinates(syllable_ends,
                                                            coarse_factor),
                            syllable_starts+1,
                            coarse_length)
    return syllable_starts, syllable_ends






def get_waliji_feature_map(s,
                           log_part_blocks,
                           log_invpart_blocks,
                           abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                           spread_length=3,
                           fft_length=512,
                           num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7,
                           return_S=False,
                           return_E=False):
    """
    Input is usually just the signal s as the rest of the parameters
    are not going to change very often

    Parameters:
    ===========
    s: np.ndarray[ndim=1]
        Raw signal data that we are extracting feature from
    log_part_blocks: np.ndarray[ndim=4,dtype=np.float32]
        First dimension is over the different features
    log_invpart_blocks: np.ndarray[ndim=4,dtype=np.float32]
        Essentially the same array as log_part_blocks. Related
        by its equal to np.log(1-np.exp(log_part_blocks))
    """
    S = esp.get_spectrogram_features(s,
                                     sample_rate,
                                     num_window_samples,
                                     num_window_step_samples,
                                     fft_length,
                                     freq_cutoff,
                                     kernel_length,
                                 )
    E, edge_feature_row_breaks,\
        edge_orientations = esp._edge_map_no_threshold(S)
    esp._edge_map_threshold_segments(E,
                                 40,
                                 1,
                                 threshold=.7,
                                 edge_orientations = edge_orientations,
                                 edge_feature_row_breaks = edge_feature_row_breaks)
    E2 = reorg_part_for_fast_filtering(E.copy())
    F = cp.code_parts_fast(E2.astype(np.uint8),log_part_blocks,log_invpart_blocks,10)
    F = np.argmax(F,2)
    # the amount of spreading to do is governed by the size of the part features
    F = swp.spread_waliji_patches(F,
                                  log_part_blocks.shape[1],
                                  log_part_blocks.shape[2],
                                  log_part_blocks.shape[0])
    F = collapse_to_grid(F,log_part_blocks.shape[1],
                         log_part_blocks.shape[2])
    if not return_S and not return_E:
        return F
    else:
        return_tuple = (F,)
        if return_S:
            return_tuple += (S,)
        if return_E:
            return_tuple += (E,)

    return return_tuple

def get_feature_map(s,
                           log_part_blocks=None,
                           log_invpart_blocks=None,
                           abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                           spread_length=3,
                           fft_length=512,
                           num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7,
                           return_S=False,
                           return_E=False):
    """
    Input is usually just the signal s as the rest of the parameters
    are not going to change very often

    Parameters:
    ===========
    s: np.ndarray[ndim=1]
        Raw signal data that we are extracting feature from
    log_part_blocks: np.ndarray[ndim=4,dtype=np.float32]
        First dimension is over the different features
    log_invpart_blocks: np.ndarray[ndim=4,dtype=np.float32]
        Essentially the same array as log_part_blocks. Related
        by its equal to np.log(1-np.exp(log_part_blocks))
    """
    S = esp.get_spectrogram_features(s,
                                     sample_rate,
                                     num_window_samples,
                                     num_window_step_samples,
                                     fft_length,
                                     freq_cutoff,
                                     kernel_length,
                                 )
    E, edge_feature_row_breaks,\
        edge_orientations = esp._edge_map_no_threshold(S)
    esp._edge_map_threshold_segments(E,
                                 40,
                                 1,
                                 threshold=.7,
                                 edge_orientations = edge_orientations,
                                 edge_feature_row_breaks = edge_feature_row_breaks)
    E = E.astype(np.uint8)
    if log_part_blocks is None:
        E = reorg_part_for_fast_filtering(E)
        if not return_S:
            return E
        else:
            return S,E
    # implicitly else just says that we are doing the further E processing
    if return_E:
        E2 = reorg_part_for_fast_filtering(E.copy())
        F = cp.code_parts_fast(E2,log_part_blocks,log_invpart_blocks,10)
    else:
        E = reorg_part_for_fast_filtering(E)
        F = cp.code_parts_fast(E,log_part_blocks,log_invpart_blocks,10)
    F = np.argmax(F,2)
    # the amount of spreading to do is governed by the size of the part features
    F = swp.spread_waliji_patches(F,
                                  log_part_blocks.shape[1],
                                  log_part_blocks.shape[2],
                                  log_part_blocks.shape[0])
    F = collapse_to_grid(F,log_part_blocks.shape[1],
                         log_part_blocks.shape[2])
    if not return_S and not return_E:
        return F
    else:
        return_tuple = (F,)
        if return_S:
            return_tuple += (S,)
        if return_E:
            return_tuple += (E,)

    return return_tuple

def _add_phns_to_phn_dict(phns,phn_dict,k):
    for i in xrange(k-1,len(phns)):
        phn_dict[tuple(
            phns[i-k+j+1]
            for j in xrange(k))] += 1

def get_ordered_kgram_phone_list(train_data_path,file_indices,k):
    phn_dict = collections.defaultdict(int)
    for data_file_idx in file_indices:
        _add_phns_to_phn_dict(
            np.load(train_data_path+data_file_idx+'phns.npy'),
            phn_dict,k)

    return sorted(phn_dict.iteritems(),key=operator.itemgetter(1),reverse=True)





def _get_syllables_examples_background_files(train_data_path,
                                            data_file_idx,
                                            avg_bgd,
                                            syllables,
                                            syllable_examples,
                                            backgrounds,
                                            log_part_blocks,
                                            log_invpart_blocks,
                                            abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02]),
                                            spread_length=3,
                                            fft_length=512,
                                            num_window_step_samples=80,
                                            freq_cutoff=3000,
                                            sample_rate=16000,
                                            num_window_samples=320,
                                            kernel_length=7,
                                            feature_type='waliji'):
    """
    Perform main signal processin
    """
    s = np.load(train_data_path+data_file_idx+'s.npy')
    phns = np.load(train_data_path+data_file_idx+'phns.npy')
    # we divide by 5 since we coarsen in the time domain
    flts = np.load(train_data_path+data_file_idx+'feature_label_transitions.npy')
    F = get_waliji_feature_map(s,
                               log_part_blocks,
                               log_invpart_blocks,
                               abst_threshold=abst_threshold,
                               spread_length=spread_length,
                               fft_length=fft_length,
                               num_window_step_samples=num_window_step_samples,
                               freq_cutoff=freq_cutoff,
                               sample_rate=sample_rate,
                               num_window_samples=num_window_samples,
                               kernel_length=kernel_length)

    background_length=3
    example_starts_ends_dict = dict(
        (syll,get_examples_from_phns_ftls(syll,
                                                               phns,
                                                               flts,
                                                               log_part_blocks,
                                                               F.shape[0]))
        for syll in syllables)


    avg_bgd.add_frames(F,time_axis=0)
    for syll, (example_starts, example_ends) in example_starts_ends_dict.items():
        if example_starts is not None:
            syllable_examples[syll].extend([F[s:e]
                                 for s,e in itertools.izip(example_starts,example_ends)])
            backgrounds[syll].extend([
                    F[max(0,s-background_length):s].mean(0)
                    for s in example_starts])



def _get_syllable_examples_background_files(train_data_path,
                                            data_file_idx,
                                            avg_bgd,
                                            syllable,
                                            syllable_examples,
                                            backgrounds=None,
                                            log_part_blocks=None,
                                            log_invpart_blocks=None,
                                            s_chunks=None,
                                            abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02]),
                                            spread_length=3,
                                            spread_type='line',
                                            fft_length=512,
                                            num_window_step_samples=80,
                                            freq_cutoff=3000,
                                            sample_rate=16000,
                                            num_window_samples=320,
                                            kernel_length=7,
                                            offset=3
                                            ):
    """
    Perform main signal processin
    """
    s = np.load(train_data_path+data_file_idx+'s.npy')
    phns = np.load(train_data_path+data_file_idx+'phns.npy')
    # we divide by 5 since we coarsen in the time domain
    flts = np.load(train_data_path+data_file_idx+'feature_label_transitions.npy')
    F = get_feature_map(s,
                        log_part_blocks,
                        log_invpart_blocks,
                        abst_threshold=abst_threshold,
                        spread_length=spread_length,
                        spread_type=spread_type,
                        fft_length=fft_length,
                        num_window_step_samples=num_window_step_samples,
                        freq_cutoff=freq_cutoff,
                        sample_rate=sample_rate,
                        num_window_samples=num_window_samples,
                        kernel_length=kernel_length,
                        use_mels=use_mels,
                        offset=offset,)
    if log_part_blocks is None:
        background_length = 20
    else:
        background_length=4* log_part_blocks.shape[1]
    example_starts, example_ends = get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               flts,
                                                               log_part_blocks,
                                                               F.shape[0])
    print "Example starts are in:", example_starts
    avg_bgd.add_frames(F,time_axis=0)
    if example_starts is not None:
        syllable_examples.extend([F[s:e]
                                 for s,e in itertools.izip(example_starts,example_ends)])
        if backgrounds is not None:
            backgrounds.extend([
                    F[max(0,s-background_length):s].mean(0)
                    for s in example_starts])


def get_syllable_examples_backgrounds_files(train_data_path,
                                            data_files_indices,
                                            syllable,
                                            log_part_blocks=None,
                                            log_invpart_blocks=None,
                                            num_examples=-1,
                                            verbose=False,
                                            return_backgrounds=True,
                                            offset=0,
                                            return_s=False):
    avg_bgd = AverageBackground()
    syllable_examples = []
    if num_examples == -1:
        num_examples = len(data_files_indices)
    if return_backgrounds:
        backgrounds = []
    else:
        backgrounds = None
    if return_s:
        s_chunks = []
    else:
        s_chunks = None
    for i,data_file_idx in enumerate(data_files_indices[:num_examples]):
        if verbose:
            if ((i % verbose) == 0 ):
                print "Getting examples from example %d" % i

        _get_syllable_examples_background_files(train_data_path,
                                                 data_file_idx,
                                                 avg_bgd,
                                                syllable,
                                                 syllable_examples,
                                                 backgrounds,
                                                 log_part_blocks,
                                                 log_invpart_blocks,
                                                s_chunks=s_chunks)
    return avg_bgd, syllable_examples, backgrounds

def get_syllables_examples_backgrounds_files(train_data_path,
                                            data_files_indices,
                                            syllables,
                                            log_part_blocks,
                                            log_invpart_blocks,
                                            num_examples=-1,
                                            verbose=False,
                                             feature_type='waliji'):
    avg_bgd = AverageBackground()
    syllable_examples = dict( (syll,[]) for syll in syllables)
    if num_examples == -1:
        num_examples = len(data_files_indices)
    backgrounds = dict( (syll,[]) for syll in syllables)
    for i,data_file_idx in enumerate(data_files_indices[:num_examples]):
        if verbose:
            if ((i % verbose) == 0 ):
                print "Getting examples from example %d" % i

        _get_syllables_examples_background_files(train_data_path,
                                                 data_file_idx,
                                                 avg_bgd,
                                                syllables,
                                                 syllable_examples,
                                                 backgrounds,
                                                 log_part_blocks,
                                                 log_invpart_blocks,
                                                 feature_type=feature_type)
    return avg_bgd, syllable_examples, backgrounds


def get_detect_lengths(data_files_indices,data_path,
                           abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                           spread_length=3,
                           fft_length=512,
                           num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7):
    return np.array([
            esp.get_spectrogram_features(np.load(data_path+data_file_idx+'s.npy'),
                                         sample_rate,
                                         num_window_samples,
                                         num_window_step_samples,
                                         fft_length,
                                         freq_cutoff,
                                         kernel_length,
                                         ).shape[1]
            for data_file_idx in data_files_indices])

def _save_detection_results(s,phns,flts,
                            detection_array,
                            detect_lengths,
                            linear_filter,c,
                            syllable,
                            example_start_end_times,
                            log_part_blocks=None,
                            log_invpart_blocks=None,
                            add_c = False,
                         abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                         spread_length=3,
                         fft_length=512,
                         num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7
                            ):
    """
    Detection array will have detection scores saved to it
    and we will make entries of detection_array that are trailing
    be some minimum value: min_val, which is initially None
    and if it is None then it is set to
         - 2* np.abs(detection_array[next_id,:next_length]).max()
         and then it is set to that threshold the rest of the time

    we also save the lengths to detect_lengths
    """
    F = get_feature_map(s,
                               log_part_blocks,
                               log_invpart_blocks,
                               abst_threshold=abst_threshold,
                               spread_length=spread_length,
                               fft_length=fft_length,
                               num_window_step_samples=num_window_step_samples,
                               freq_cutoff=freq_cutoff,
                               sample_rate=sample_rate,
                               num_window_samples=num_window_samples,
                               kernel_length=kernel_length)


    detect_lengths.append(F.shape[0])


    example_starts, example_ends = get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               flts,
                                                               log_part_blocks,
                                                               F.shape[0])

    detection_array[len(detect_lengths)-1,
                    :F.shape[0]-linear_filter.shape[0]+1] = compute_likelihood_linear_filter.detect(F.astype(np.uint8),
                                                                                             linear_filter)
    if add_c:
        detection_array[len(detect_lengths)-1,
                    :F.shape[0]-linear_filter.shape[0]+1] += c

    if example_starts is not None:
        example_start_end_times.append(zip(example_starts,example_ends))
    else:
        example_start_end_times.append([])


def _save_detection_results_mixture(s,phns,flts,
                            detection_array,
                            detect_lengths,
                            linear_filters_cs,
                            syllable,
                            example_start_end_times,
                            log_part_blocks=None,
                            log_invpart_blocks=None,
                         abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                         spread_length=3,
                         fft_length=512,
                         num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7
                            ):
    """
    Detection array will have detection scores saved to it
    and we will make entries of detection_array that are trailing
    be some minimum value: min_val, which is initially None
    and if it is None then it is set to
         - 2* np.abs(detection_array[next_id,:next_length]).max()
         and then it is set to that threshold the rest of the time

    we also save the lengths to detect_lengths
    """
    F = get_feature_map(s,
                               log_part_blocks,
                               log_invpart_blocks,
                               abst_threshold=abst_threshold,
                               spread_length=spread_length,
                               fft_length=fft_length,
                               num_window_step_samples=num_window_step_samples,
                               freq_cutoff=freq_cutoff,
                               sample_rate=sample_rate,
                               num_window_samples=num_window_samples,
                               kernel_length=kernel_length)


    detect_lengths.append(F.shape[0])


    example_starts, example_ends = get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               flts,
                                                               log_part_blocks,
                                                               F.shape[0])


    detection_array[len(detect_lengths)-1,
                    :F.shape[0]-linear_filters_cs[0][0].shape[0]+1] = compute_likelihood_linear_filter.detect(F.astype(np.uint8),
                                                                                             linear_filters_cs[0][0]) + linear_filters_cs[0][1]
    if len(linear_filters_cs) > 1:
        for cur_filt,cur_c in linear_filters_cs[1:]:
            detection_array[len(detect_lengths)-1,
                    :F.shape[0]-cur_filt.shape[0]+1] = np.maximum(compute_likelihood_linear_filter.detect(F.astype(np.uint8),
                                                                                             cur_filt) + cur_c,
                                                                                             detection_array[len(detect_lengths)-1,
                    :F.shape[0]-cur_filt.shape[0]+1])


    if example_starts is not None:
        example_start_end_times.append(zip(example_starts,example_ends))
    else:
        example_start_end_times.append([])




def get_detection_scores(data_path,
                         detection_array,
                         syllable,
                         linear_filter,c,
                         log_part_blocks=None,
                         log_invpart_blocks=None,
                         abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                         spread_length=3,
                         fft_length=512,
                         num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7,
                         verbose = False,
                         num_examples =-1):
    data_files_indices = get_data_files_indices(data_path)

    example_start_end_times = []
    if num_examples == -1:
        num_examples = len(data_files_indices)

    detection_lengths = []
    for i,data_file_idx in enumerate(data_files_indices[:num_examples]):
        if verbose:
            if ((i % verbose) == 0 ):
                print "Getting examples from example %d" % i

        s = np.load(data_path+data_file_idx+'s.npy')
        phns = np.load(data_path+data_file_idx+'phns.npy')
        # we divide by 5 since we coarsen in the time domain
        flts = np.load(data_path+data_file_idx+'feature_label_transitions.npy')


        _save_detection_results(s,phns,flts,
                                detection_array,
                                detection_lengths,
                                linear_filter,c,
                                syllable,
                                example_start_end_times,
                                log_part_blocks,
                                log_invpart_blocks
                                )
    return detection_array,example_start_end_times, detection_lengths


def get_detection_scores_mixture(data_path,
                         detection_array,
                         syllable,
                         linear_filters_cs,
                         log_part_blocks=None,
                         log_invpart_blocks=None,
                         abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                         spread_length=3,
                         fft_length=512,
                         num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7,
                         verbose = False,
                         num_examples =-1):
    data_files_indices = get_data_files_indices(data_path)

    example_start_end_times = []
    if num_examples == -1:
        num_examples = len(data_files_indices)

    detection_lengths = []
    for i,data_file_idx in enumerate(data_files_indices[:num_examples]):
        if verbose:
            if ((i % verbose) == 0 ):
                print "Getting examples from example %d" % i

        s = np.load(data_path+data_file_idx+'s.npy')
        phns = np.load(data_path+data_file_idx+'phns.npy')
        # we divide by 5 since we coarsen in the time domain
        flts = np.load(data_path+data_file_idx+'feature_label_transitions.npy')


        _save_detection_results_mixture(s,phns,flts,
                                detection_array,
                                detection_lengths,
                                linear_filters_cs,
                                syllable,
                                example_start_end_times,
                                log_part_blocks,
                                log_invpart_blocks
                                )
    return detection_array,example_start_end_times, detection_lengths

def get_detection_scores_mixture_hack(data_path,
                         detection_array,
                         syllable,
                         linear_filters_cs,
                         log_part_blocks=None,
                         log_invpart_blocks=None,
                         abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                         spread_length=3,
                         fft_length=512,
                         num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7,
                         verbose = False,
                         num_examples =-1):
    detection_array[:] = -np.inf
    print syllable
    print linear_filters_cs
    for i in xrange(len(linear_filters_cs)):
        linear_filter = linear_filters_cs[i][0]
        c = linear_filters_cs[i][1]

        new_detection_array = np.zeros(detection_array.shape,
                                       dtype=detection_array.dtype)
        #import pdb; pdb.set_trace()
        new_detection_array,example_start_end_times, detection_lengths = get_detection_scores(data_path,
                                                                                                       new_detection_array,
                                                                                                       syllable,
                                                                                                       linear_filter,c,
                                                                                                       log_part_blocks,
                                                                                                       log_invpart_blocks,verbose=True)

        detection_array = np.maximum(new_detection_array+c,detection_array).astype(detection_array.dtype)

    return detection_array,example_start_end_times, detection_lengths



def get_training_examples(syllable,train_data_path):
    data_files_indices = get_data_files_indices(train_data_path)



FeatureMapTimeMap = collections.namedtuple("FeatureMapTimeMap",
                                           ("name"
                                            +" feature_map"
                                            +" time_map"
                                            +" parameter_map"))


Utterance = collections.namedtuple("Utterance",
                                    ("utterance_directory"
                                    +" file_idx"
                                    +" s"
                                    +" phns"
                                    +" flts"
                                    ))


def makeUtterance(
                 utterance_directory,
                 file_idx):
    return Utterance(
        utterance_directory=utterance_directory,
        file_idx=file_idx,
        s = np.load(utterance_directory+file_idx+'s.npy'),
        phns = np.load(utterance_directory+file_idx+'phns.npy'),
        flts = np.load(utterance_directory+file_idx+'feature_label_transitions.npy'))


def makeTimeMap(old_start,old_end,new_start,new_end):
    return (np.arange(new_start,new_end) * float(old_end-old_start)/float(new_end-new_start) + .5).astype(int)


SyllableFeatures = collections.namedtuple("SyllableFeatures",
                                          "s S S_config E E_config offset")


def phns_syllable_matches(phns,syllable):
    syllable_len = len(syllable)
    return np.array([ phn_id
                      for phn_id in xrange(len(phns)-syllable_len+1)
                      if np.all(phns[phn_id:phn_id+syllable_len]==np.array(syllable))])

def get_example_with_offset(F,offset,start_idx,end_idx):
    return np.vstack(
        (np.inf * np.ones((-min(start_idx-offset,0),)+F.shape[1:]),
         F[start_idx:end_idx],
         np.inf * np.ones((-min(end_idx-offset,0),) + F.shape[1:])))

def get_syllable_features(utterance_directory,data_idx,syllable,
                          S_config=None,E_config=None,offset = None,E_verbose=False,avg_bgd=None):
    """
    Expects a list of (name, parameters) tuples
    names are:
       waveform
       spectrogram
       edge_map
       waliji
    """
    utterance = makeUtterance(utterance_directory,data_idx)
    sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
    # get the spectrogram
    if S_config is not None:
        S = get_spectrogram(utterance.s,S_config)
        S_flts = (sflts * S.shape[0] /float(sflts[-1]) + .5).astype(int)
        if E_config is not None:
            E = get_edge_features(S.T,E_config,verbose=E_verbose)
            if avg_bgd is not None:
                avg_bgd.add_frames(E,time_axis=0)
            # both are the same
            E_flts = S_flts
        else:
            E = None
    else:
        S = None
        E = None
    # we then get the example phones removed from the signals
    syllable_starts = phns_syllable_matches(utterance.phns,syllable)
    syllable_length = len(syllable)
    if (offset is None or offset == 0) and (S is not None and E is not None):
        return [ SyllableFeatures(
                s = (utterance.s)[sflts[syllable_start]:sflts[syllable_start+syllable_length]],
                S = S[S_flts[syllable_start]:S_flts[syllable_start+syllable_length]],
                S_config = S_config,
                E = E[E_flts[syllable_start]:E_flts[syllable_start+syllable_length]],
                E_config = E_config,
                offset = 0)
                 for syllable_start in syllable_starts]
    elif (S is not None and E is not None):
        return [ SyllableFeatures(
                s = (utterance.s)[sflts[syllable_start]:sflts[syllable_start+syllable_length]],
                S = get_example_with_offset(S,offset,S_flts[syllable_start],S_flts[syllable_start+syllable_length]),
                S_config = S_config,
                E = get_example_with_offset(E,offset,E_flts[syllable_start],E_flts[syllable_start+syllable_length]),
                E_config = E_config,
                offset = offset)
                 for syllable_start in syllable_starts]
    else:
        return None




def get_syllable_features_directory(utterances_path,file_indices,syllable,
                                    S_config=None,E_config=None,offset=None,
                                    E_verbose=False,return_avg_bgd=True):
    avg_bgd = AverageBackground()
    return_tuple = tuple(
        get_syllable_features(utterances_path,data_idx,syllable,
                              S_config=S_config,E_config=E_config,offset = offset,
                              E_verbose=E_verbose,avg_bgd=avg_bgd)
        for data_idx in file_indices)
    if return_avg_bgd:
        return return_tuple, avg_bgd
    else:
        return return_tuple

def recover_example_map(syllable_features):
    return np.array(reduce(lambda x,y : x + y,
                           [[i] * len(e)
                            for i,e in enumerate(syllable_features)]))

def recover_edgemaps(syllable_features,example_mat):
    max_length = max(
        max((0,) + tuple(s.E.shape[0] for s in e))
        for e in syllable_features)
    for e in syllable_features:
        if len(e) > 0: break
    E_shape = e[0].E.shape[1:]
    Es = np.zeros((len(example_mat),max_length)+E_shape)
    lengths = np.zeros(len(example_mat))
    jdx=0
    for idx,i in enumerate(example_mat):
        if idx > 0:
            if i == example_mat[idx-1]:
                jdx += 1
            else:
                jdx = 0
            
            lengths[idx] = len(syllable_features[i][jdx].E)
            Es[idx][:lengths[idx]] = syllable_features[i][jdx].E
        else:
            lengths[idx] = len(syllable_features[i][jdx].E)
            Es[idx][:lengths[idx]] = syllable_features[i][jdx].E
    return lengths, Es

def recover_specs(syllable_features,example_mat):
    max_length = max(
        max((0,) + tuple(s.S.shape[0] for s in e))
        for e in syllable_features)
    for e in syllable_features:
        if len(e) > 0: break
    S_shape = e[0].S.shape[1:]
    Ss = np.zeros((len(example_mat),max_length)+S_shape)
    lengths = np.zeros(len(example_mat))
    jdx=0
    for idx,i in enumerate(example_mat):
        if idx > 0:
            if i == example_mat[idx-1]:
                jdx += 1
            else:
                jdx = 0
            
            lengths[idx] = len(syllable_features[i][jdx].S)
            Ss[idx][:lengths[idx]] = syllable_features[i][jdx].S
        else:
            lengths[idx] = len(syllable_features[i][jdx].S)
            Ss[idx][:lengths[idx]] = syllable_features[i][jdx].S
    return lengths, Ss


def recover_waveforms(syllable_features,example_mat):
    max_length = max(
        max((0,) + tuple(s.s.shape[0] for s in e))
        for e in syllable_features)
    waveforms = np.zeros((len(example_mat),max_length))
    lengths = np.zeros(len(example_mat))
    jdx=0
    for idx,i in enumerate(example_mat):
        if idx > 0:
            if i == example_mat[idx-1]:
                jdx += 1
            else:
                jdx = 0
            
            lengths[idx] = len(syllable_features[i][jdx].s)
            waveforms[idx][:lengths[idx]] = syllable_features[i][jdx].s
        else:
            lengths[idx] = len(syllable_features[i][jdx].s)
            waveforms[idx][:lengths[idx]] = syllable_features[i][jdx].s
    return lengths, waveforms


SpectrogramParameters = collections.namedtuple("SpectrogramParameters",
                                               ("sample_rate"
                                                +" num_window_samples"
                                                +" num_window_step_samples"
                                                +" fft_length"
                                                +" kernel_length"
                                                +" freq_cutoff"
                                                +" use_mel"))

def get_spectrogram(waveform,spectrogram_parameters):
    if spectrogram_parameters.use_mel:
        return get_mel_spec(waveform,
                            spectrogram_parameters.sample_rate,
                        spectrogram_parameters.num_window_samples,
                        spectrogram_parameters.num_window_step_samples,
                        spectrogram_parameters.fft_length).T
    else:
        return esp.get_spectrogram_features(waveform,
                                     spectrogram_parameters.sample_rate,
                                     spectrogram_parameters.num_window_samples,
                                     spectrogram_parameters.num_window_step_samples,
                                     spectrogram_parameters.fft_length,
                                     spectrogram_parameters.freq_cutoff,
                                     spectrogram_parameters.kernel_length,
                                 ).T

EdgemapParameters = collections.namedtuple("EdgemapParameters",
                                           ("block_length"
                                            +" spread_length"
                                            +" threshold"))


def get_edge_features(S,parameters,verbose=False):
    E, edge_feature_row_breaks,\
        edge_orientations = esp._edge_map_no_threshold(S)
    esp._edge_map_threshold_segments(E,
                                     parameters.block_length,
                                     parameters.spread_length,
                                     threshold=parameters.threshold,
                                     edge_orientations = edge_orientations,
                                     edge_feature_row_breaks = edge_feature_row_breaks,
                                     verbose=verbose)
    return reorg_part_for_fast_filtering(E)

