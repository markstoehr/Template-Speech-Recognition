import numpy as np
from scipy import ndimage

import sys, os, cPickle,re,itertools,collections,operator

import bernoulli_mixture as bernoulli_mixture
import edge_signal_proc as esp
import extract_local_features as elf
import code_parts as cp
import spread_waliji_patches as swp
import compute_likelihood_linear_filter
import multiprocessing

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
                                coarse_length,
                                verbose=False):
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
    phn_matches = phns_syllable_matches(phns,syllable)
    if verbose:
        print phn_matches
        print syllable
        print phns
    if phn_matches.shape[0] ==0:
        return None, None
    syllable_starts = flts[ phn_matches]
    syllable_ends = flts[phn_matches + len(syllable)]

    if verbose:
        print syllable_starts
        print syllable_ends

    if log_part_blocks is not None:
        coarse_factor = log_part_blocks.shape[1]
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

def get_classify_lengths(data_files_indices,data_path,
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
            np.load(data_path+data_file_idx+'phns.npy').shape[0]
            for data_file_idx in data_files_indices])

def get_classify_labels(data_files_indices,data_path,classify_lengths,
                           abst_threshold=np.array([.025,.025,.015,.015,
                                                     .02,.02,.02,.02]),
                           spread_length=3,
                           fft_length=512,
                           num_window_step_samples=80,
                           freq_cutoff=3000,
                           sample_rate=16000,
                           num_window_samples=320,
                           kernel_length=7):
    classify_labels = np.empty((len(classify_lengths),
                                classify_lengths.max()),
                                dtype='|S4')
    for utt_id, data_file_idx in enumerate(data_files_indices):
        phns = np.load(data_path+data_file_idx+'phns.npy')
        classify_labels[utt_id][:classify_lengths[utt_id]] = (
            phns[:])
        if not np.all(classify_labels[utt_id][:classify_lengths[utt_id]] == phns[:]):
            import pdb; pdb.set_trace()
    return classify_labels


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

    if np.any(np.isnan(detection_array)) == True:
        import pdb; pdb.set_trace()

    if example_starts is not None:
        example_start_end_times.append(zip(example_starts,example_ends))
    else:
        example_start_end_times.append([])


def compute_detection_E(E,phns,E_flts,
                        detection_array,
                        cascade,
                         syllable,
                         phn_mapping=None,
                         verbose=False):
    """
    Detection array will have detection scores saved to it
    and we will make entries of detection_array that are trailing
    be some minimum value: min_val, which is initially None
    and if it is None then it is set to
         - 2* np.abs(detection_array[next_id,:next_length]).max()
         and then it is set to that threshold the rest of the time

    we also save the lengths to detect_lengths
    """
    example_starts, example_ends = get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               E_flts,
                                                               None,
                                                               E.shape[0],
                                                               verbose=verbose)


    # detection_template_ids is either None or Zero so we are set

    filter_id = -1
    for cur_filt,cur_c in cascade:
        filter_id += 1
        detect_array[filter_id] = compute_likelihood_linear_filter.detect(E.astype(np.uint8),
                                                    cur_filt) + cur_c

    return E_length-1,zip(example_starts,example_ends)

def get_classify_windows(phns,flts):
    """
    Compute the windows over which we get the classification scores
    """
    window_starts = np.maximum(0,flts[:-1] + (flts[:-1] - flts[1:])/3.).astype(np.int16)
    window_ends = np.minimum(flts[-1],flts[:-1] + (flts[1:]-flts[:-1])/3.).astype(np.int16)
    return window_starts, window_ends

def _compute_classification_E(E,phns,E_flts,
                              utt_id,
                         classify_array,
                         classify_template_ids,
                         classify_template_lengths,
                         classify_locs,
                         linear_filters_cs,
                         phn_mapping=None,
                         verbose=False
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
    window_starts, window_ends = get_classify_windows(phns,E_flts)

    detection_scores  = (compute_likelihood_linear_filter.detect(
                            E.astype(np.uint8),
                            linear_filters_cs[0][0])
                       + linear_filters_cs[0][1])
    filter_id = 0

    for phn_id, window_start_end in enumerate(itertools.izip(window_starts,
                                                             window_ends)):
        w_start, w_end = window_start_end
        classify_locs[utt_id,phn_id] = w_start + np.argmax(detection_scores[w_start:w_end])
        classify_template_ids[utt_id,phn_id] = filter_id

        classify_template_lengths[utt_id,phn_id] = len(linear_filters_cs[0][0])

        classify_array[utt_id,phn_id] =  detection_scores[classify_locs[utt_id,phn_id]]



    if len(linear_filters_cs) > 1:
        for cur_filt,cur_c in linear_filters_cs[1:]:
            filter_id += 1
            detection_scores = compute_likelihood_linear_filter.detect(E.astype(np.uint8),
                                                                                             cur_filt) + cur_c

            # check if we get a higher score for any of these objects
            for phn_id, window_start_end in enumerate(itertools.izip(window_starts,
                                                             window_ends)):
                w_start, w_end = window_start_end
                if len(detection_scores[w_start:w_end]) < 1:
                    import pdb; pdb.set_trace()
                cur_filter_classify_time = w_start + np.argmax(detection_scores[w_start:w_end])
                if classify_array[utt_id,phn_id] <  detection_scores[cur_filter_classify_time]:
                    classify_locs[utt_id,phn_id] = cur_filter_classify_time
                    classify_template_ids[utt_id,phn_id] = filter_id
                    classify_template_lengths[utt_id,phn_id] = len(cur_filt)
                    classify_array[utt_id,phn_id] =  detection_scores[classify_locs[utt_id,phn_id]]


    if np.any(np.isnan(classify_array)) == True:
        import pdb; pdb.set_trace()

def get_isolated_classify_windows(E,phns,flts,bgd,linear_filters_cs):
    """
    Compute the windows over which we get the classification scores
    """
    # get the maximum filter length
    max_filter_length = max( len(lfc[0]) for lfc in linear_filters_cs)
    max_phn_length_third = int(np.ceil(np.max(flts[1:]-flts[:-1])/3.))
    # prepend the beginning of E with background for padding purposes
    E = np.vstack((
        np.tile(bgd ,
                (max_filter_length+max_phn_length_third,)
                +tuple(
                    np.ones(
                        len(
                            bgd.shape)
                            )
                            )
                            ).astype(np.float32),
                E.astype(np.float32),
                   np.tile(bgd,(max(0,flts[-2]+max_phn_length_third+max_filter_length-flts[-1]),)
                           +tuple(np.ones(len(bgd.shape)))).astype(np.float32)))
    # append to the end of E with enough padding for maximum filter length
    flts += max_filter_length+max_phn_length_third
    window_starts = (flts[:-1] + (flts[:-1] - flts[1:])/3.).astype(np.int16)
    window_ends = (flts[:-1] + (flts[1:]-flts[:-1])/3.).astype(np.int16)
    # we get from the very start of where we can check
    # and make sure that the maximum filter isn't greater than the max_filter
    # length
    # phn_segment_ends correspond to the end of the segment into the utterance that we count

    phn_segment_ends = ( flts[1:] + (flts[1:]-flts[:-1])/3.).astype(np.int16)

    # we stack background onto the end of the E_window
    # so that we don't get too much of the following phone
    # in the phone being analyzed
    E_windows = tuple(
        np.vstack((
            E[window_start:phn_segment_end].astype(np.float32),
            np.tile(bgd,
                    (max(0,window_end+max_filter_length - phn_segment_end),)
                    +tuple(np.ones(len(bgd.shape))))))
        for window_start, window_end, phn_segment_end in itertools.izip(window_starts,
                                                       window_ends,
                                                       phn_segment_ends))
    if np.any(window_starts <0):
        import pdb; pdb.set_trace()
    # we return both the windows in terms of features
    # and we return the start locations of the windows
    return E_windows, window_starts, window_ends, flts[0]


def _compute_isolated_classification_E(E,phns,E_flts,
                              utt_id,
                         classify_array,
                         classify_template_ids,
                         classify_template_lengths,
                         classify_locs,
                         linear_filters_cs,
                                       bgd,
                         phn_mapping=None,
                         verbose=False,
                                       svm_classifiers=None,
                                       svm_constants=None,
                                       svm_classify_array=None
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
    if utt_id % 200 ==0:
        print phns, E_flts, utt_id

    E_windows, window_starts, window_ends, flt_front_pad = get_isolated_classify_windows(E,phns,E_flts,bgd,linear_filters_cs)
    for phn_id, E_window_p_starts_p_ends in enumerate(itertools.izip(
            E_windows,window_starts,
            window_ends)):

        E_window, w_start, w_end = E_window_p_starts_p_ends

        detection_scores  = (compute_likelihood_linear_filter.detect_float(
                            E_window,
                            linear_filters_cs[0][0])
                             + linear_filters_cs[0][1])
        filter_id = 0

        classify_locs[utt_id,phn_id] = w_start + np.argmax(detection_scores[:w_end-w_start]) -flt_front_pad
        classify_template_ids[utt_id,phn_id] = filter_id

        classify_template_lengths[utt_id,phn_id] = len(linear_filters_cs[0][0])

        classify_array[utt_id,phn_id] =  detection_scores[classify_locs[utt_id,phn_id]-w_start+flt_front_pad]
        for cur_filt,cur_c in linear_filters_cs[1:]:
            filter_id += 1

            detection_scores = compute_likelihood_linear_filter.detect_float(E_window,
                                                                                             cur_filt) + cur_c
            cur_filter_classify_time = w_start + np.argmax(detection_scores[:w_end-w_start]) - flt_front_pad

            if classify_array[utt_id,phn_id] <  detection_scores[cur_filter_classify_time-w_start]:
                classify_locs[utt_id,phn_id] = cur_filter_classify_time
                classify_template_ids[utt_id,phn_id] = filter_id

                classify_template_lengths[utt_id,phn_id] = len(cur_filt)
                classify_array[utt_id,phn_id] =  detection_scores[classify_locs[utt_id,phn_id]-w_start+flt_front_pad]


        # now we run the phone through the svm filter for the best mixture
        # component
        if (svm_classifiers is not None) and (svm_classify_array is not None):
            svm_classify_array[utt_id,phn_id] = (E_window[
                classify_locs[utt_id,phn_id] - w_start+flt_front_pad:
                    classify_locs[utt_id,phn_id] - w_start
                +flt_front_pad
                + classify_template_lengths[utt_id,phn_id]] * svm_classifiers[classify_template_ids[utt_id,phn_id]]).sum() + svm_constants[classify_template_ids[utt_id,phn_id]]

    import pdb; pdb.set_trace()
    if np.any(np.isnan(classify_array)) == True:
        import pdb; pdb.set_trace()




def _compute_detection_E(E,phns,E_flts,
                         detection_array,
                         detection_template_ids,
                         detect_lengths,
                         linear_filters_cs,
                         syllable,
                         example_start_end_times,
                         phn_mapping=None,
                         verbose=False
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
    detect_length = 0




    example_starts, example_ends = get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               E_flts,
                                                               None,
                                                               E.shape[0],
                                                               verbose=verbose)

    if len(detect_lengths) % 200 == 0:
        print "len(detect_lengths)=%d" % (len(detect_lengths))
    detection_array[len(detect_lengths),
                    :E.shape[0]]\
                    = (compute_likelihood_linear_filter.detect(
                            E.astype(np.uint8),
                            linear_filters_cs[0][0])
                       + linear_filters_cs[0][1])

    # detection_template_ids is either None or Zero so we are set

    filter_id = 0
    if len(linear_filters_cs) > 1:
        for cur_filt,cur_c in linear_filters_cs[1:]:
            detect_length = E.shape[0]
            filter_id += 1
            v = compute_likelihood_linear_filter.detect(E.astype(np.uint8),
                                                                                             cur_filt) + cur_c

            if detection_template_ids is not None:
                detection_template_ids[len(detect_lengths),
                                       :E.shape[0]]\
                                       [v >
                                        detection_array[
                                               len(detect_lengths),
                                               :E.shape[0]]] = filter_id

            detection_array[len(detect_lengths),
                    :E.shape[0]] = np.maximum(
                v,
                detection_array[
                            len(detect_lengths),
                            :E.shape[0]])


    if np.any(np.isnan(detection_array)) == True:
        import pdb; pdb.set_trace()

    if example_starts is not None:
        example_start_end_times.append(zip(example_starts,example_ends))
        for estart, eend in itertools.izip(example_starts,example_ends):
            if np.max(detection_array[len(detect_lengths),
                                      max(estart-11,0):min(estart+11,E.shape[0])]) == 0.:
                import pdb; pdb.set_trace()

    else:
        example_start_end_times.append([])

    detect_lengths.append(detect_length)

    if len(detect_lengths) % 100==0:
        print "E length = %d, detect_length = %d" % (E.shape[0],detect_length)




def _compute_detection_S(S,phns,S_flts,
                         detection_array,
                         detection_template_ids,
                         detect_lengths,
                         linear_filters_cs,
                         syllable,
                         example_start_end_times,
                         phn_mapping=None,
                         verbose=False):
    """
    Detection code for when we are using spectral templates
    as a baseline for the edge features

    Detection array will have detection scores saved to it
    and we will make entries of detection_array that are trailing
    be some minimum value: min_val, which is initially None
    and if it is None then it is set to
         - 2* np.abs(detection_array[next_id,:next_length]).max()
         and then it is set to that threshold the rest of the time

    we also save the lengths to detect_lengths

    Parameters:
    ===========
    """
    detect_length = 0




    example_starts, example_ends = get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               S_flts,
                                                               None,
                                                               S.shape[0],
                                                               verbose=verbose)

    num_frames = S.shape[0]
    detect_length = num_frames
    S = S.astype(np.float32)
    S_sq = S**2
    if len(detect_lengths) % 100 == 0:
        print "len(detect_lengths)=%d" % (len(detect_lengths))
    detection_array[len(detect_lengths),
                    :num_frames]\
                    = (compute_likelihood_linear_filter.detect_spectral(
                            S_sq,
                            linear_filters_cs[0][0])
                       + compute_likelihood_linear_filter.detect_spectral(
                            S,
                            linear_filters_cs[0][1])
                       + linear_filters_cs[0][2])

    # detection_template_ids is either None or Zero so we are set

    filter_id = 0
    if len(linear_filters_cs) > 1:
        for cur_sq_filt,cur_lin_filt,cur_c in linear_filters_cs[1:]:
            detect_length = num_frames
            filter_id += 1
            v = (compute_likelihood_linear_filter.detect_spectral(
                            S_sq,
                            cur_sq_filt)
                       + compute_likelihood_linear_filter.detect_spectral(
                            S,
                            cur_lin_filt)
                       + cur_c)

            if detection_template_ids is not None:
                detection_template_ids[len(detect_lengths),
                                       :detect_length]\
                                       [v >
                                        detection_array[
                                               len(detect_lengths),
                                               :detect_length]] = filter_id

            detection_array[len(detect_lengths),
                    :detect_length] = np.maximum(
                v,
                detection_array[
                            len(detect_lengths),
                            :detect_length])


    if np.any(np.isnan(detection_array)) == True:
        import pdb; pdb.set_trace()

    if example_starts is not None:
        example_start_end_times.append(zip(example_starts,example_ends))
        for estart, eend in itertools.izip(example_starts,example_ends):
            try:
                if np.max(detection_array[len(detect_lengths),
                                          max(estart-11,0):min(estart+11,detect_length)]) == 0.:
                    import pdb; pdb.set_trace()
            except:
                import pdb;pdb.set_trace()

    else:
        example_start_end_times.append([])

    detect_lengths.append(detect_length)

    if len(detect_lengths) % 100 == 0:
        print "S length = %d, detect_length = %d" % (S.shape[0],detect_length)


def get_vals_pad(x,idx,default_val,window_half_length):
    hi = len(x)
    lo_idx = idx - window_half_length
    hi_idx = idx + window_half_length + 1
    left = np.empty(max(-lo_idx,0),dtype=x.dtype)
    right = np.empty(max(hi_idx-hi,0),dtype=x.dtype)
    if len(left) > 0:
        left[:] = default_val
    if len(right) > 0:
        right[:] = default_val
    out =  np.hstack((
            left,
             x[max(0,lo_idx):
                   min(hi,hi_idx)],
            right))






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
        if np.any(np.isnan(detection_array)) == True:
            import pdb; pdb.set_trace()


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

def get_detection_scores_mixture_named_params(data_path,file_indices,
                         detection_array,
                         syllable,
                         linear_filters_cs,S_config=None,
                                              E_config=None,
                         verbose = False,
                         num_examples =-1,
                                              return_detection_template_ids=False,
                                              log_parts=None,
                                              log_invparts=None,
                                              count_threshold=None,
                                              spread_factor=1,
                                              subsamp_factor=1,
                                              phn_mapping=None,
                                              P_config=None,
                                              use_noise_file=None,
                                              noise_db=0,
                                              use_spectral=False):

    example_start_end_times = []
    if num_examples == -1:
        num_examples = len(file_indices)

    detection_lengths = []
    detection_lengths2 = []
    if return_detection_template_ids:
        detection_template_ids = np.zeros(detection_array.shape,dtype=int)
    else:
        detection_template_ids = None
    for i,data_file_idx in enumerate(file_indices[:num_examples]):
        if verbose:
            if ((i % verbose) == 0 ):
                print "Getting examples from example %d" % i


        utterance = makeUtterance(data_path,data_file_idx,
                                  use_noise_file=use_noise_file,
                                  noise_db=noise_db)
        sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
        S = get_spectrogram(utterance.s,S_config)
        S_flts = utterance.flts
        E = get_edge_features(S.T,E_config,verbose=False)


        if use_spectral:
            _compute_detection_S(S,utterance.phns,S_flts,
                                 detection_array,
                                 detection_template_ids,
                                 detection_lengths,
                                 linear_filters_cs,
                                 syllable,
                                 example_start_end_times,
                                 phn_mapping=phn_mapping,
                                 verbose=verbose
                                )

        else:
            E_flts = S_flts
            if P_config is not None:
                # E is now the part features
                E = get_part_features(E,P_config,verbose=False)
                E_flts = S_flts.copy()
                E_flts[-1] = len(E)

            _compute_detection_E(E,utterance.phns,E_flts,
                                 detection_array,
                                 detection_template_ids,
                                 detection_lengths,
                                 linear_filters_cs,
                                 syllable,
                                 example_start_end_times,
                                 phn_mapping=phn_mapping,
                                 verbose=verbose
                                 )


        if len(linear_filters_cs) ==1:
            detection_lengths2.append(len(S)-linear_filters_cs[0][0].shape[0])

    if len(linear_filters_cs) == 1:
        detection_lengths = np.array(detection_lengths2)
    else:
        detection_lengths = np.array(detection_lengths)
    if return_detection_template_ids:
        return (detection_array,
                example_start_end_times,
                detection_lengths,
                detection_template_ids)
    else:
        return (detection_array,
                example_start_end_times,
                detection_lengths)

def get_classify_scores(data_path,file_indices,
                        classify_array,
                        linear_filters_cs,
                        bgd,
                        S_config=None,
                                              E_config=None,
                        verbose = False,
                         num_examples =-1,
                        return_detection_template_ids=False,
                                              log_parts=None,
                                              log_invparts=None,
                                              count_threshold=None,
                                              spread_factor=1,
                                              subsamp_factor=1,
                                              phn_mapping=None,
                                              P_config=None,
                                              use_noise_file=None,
                                              noise_db=0,
                                              use_spectral=False,
                        svm_classifiers=None,
                        svm_constants=None):


    if svm_classifiers is None:
        svm_classify_array = None
    else:
        svm_classify_array = classify_array.copy().astype(np.float32)

    if num_examples == -1:
        num_examples = len(file_indices)

    detection_lengths2 = []

    classify_template_ids = np.zeros(classify_array.shape,dtype=np.uint16)


    classify_locs = np.zeros(classify_array.shape,dtype=np.uint16)
    classify_template_lengths = np.zeros(classify_array.shape,dtype=np.uint16)
    for utt_id,data_file_idx in enumerate(file_indices[:num_examples]):
        if verbose:
            if ((utt_id % verbose) == 0 ):
                print "Getting examples from example %d" % i


        utterance = makeUtterance(data_path,data_file_idx,
                                  use_noise_file=use_noise_file,
                                  noise_db=noise_db)
        sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
        S = get_spectrogram(utterance.s,S_config)
        S_flts = utterance.flts
        E = get_edge_features(S.T,E_config,verbose=False)


        if use_spectral:
            _compute_detection_S(S,utterance.phns,S_flts,
                                 detection_array,
                                 detection_template_ids,
                                 detection_lengths,
                                 linear_filters_cs,
                                 (),
                                 example_start_end_times,
                                 phn_mapping=phn_mapping,
                                 verbose=verbose
                                )

        else:
            E_flts = S_flts
            if P_config is not None:
                # E is now the part features
                E = get_part_features(E,P_config,verbose=False)
                E_flts = S_flts.copy()
                E_flts[-1] = len(E)

            _compute_isolated_classification_E(E,utterance.phns,E_flts,
                                      utt_id,
                                 classify_array,
                                 classify_template_ids,
                                 classify_template_lengths,
                                 classify_locs,
                                 linear_filters_cs,
                                               bgd,
                                 phn_mapping=phn_mapping,
                                 verbose=verbose,
                                               svm_classifiers=svm_classifiers,
                                               svm_constants=svm_constants,
                                               svm_classify_array=svm_classify_array
                                 )


    if svm_classifiers is None:
        return (classify_array,
                classify_locs,
                classify_template_lengths,
                classify_template_ids)
    else:
        return (classify_array,
                svm_classify_array,
                classify_locs,
                classify_template_lengths,
                classify_template_ids)
        



def get_classification_scores_mixture_named_params(data_path,file_indices,
                                                   phone_label,
                                                   linear_filters_cs,
                                                   bgd,
                                                   window_half_length,
                                                   S_config=None,
                                                   E_config=None,
                                                   verbose = False,
                                                   num_examples =-1,
                                                   return_classification_template_ids=False):
    """
    Assessing the classification performance of the dataset on my
    data.  We implement a two-stage classification, this stage is
    purely to get an ROC curve and sketch the performance basically of
    our classifier.

    The next stage is to get fine-grained statistics for praticular
    points on the ROC curve (otherwise we use up too much memory).

    We rely on a background model bgd in order to pad examples because
    the template we are using and the data do not necessarily match each
    other.

    Window half length is there because we do not totally trust the timit
    labeling.

    For each example of the phone we want some data about the detection
    that will allow us to understand the classifier we are using and get
    some insights. Candidate metadata for the detection would be:
      - file idx (for provenance purposes
      - phn labels immediately preceding
      - phn labels immediately following
      - within a window what were the detection scores and the max score
      - within a window, which were the templates that fired the hardest
        and which was the max

    We create a default dict to hold the phone labels

    we'll get this for each phn.  The idea will be to find the hard
    false positives and then we'll do a second stage detector based on
    the svm. But first we'll get some basic statistics on how this is working

    """

    phn_classification_dict = collections.defaultdict(tuple)


    for i,data_file_idx in enumerate(file_indices[:num_examples]):
        if verbose:
            if ((i % verbose) == 0 ):
                print "Getting examples from example %d" % i

        utterance = makeUtterance(data_path,data_file_idx)
        sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
        S = get_spectrogram(utterance.s,S_config)
        S_flts = (sflts * S.shape[0] /float(sflts[-1]) + .5).astype(int)
        E = get_edge_features(S.T,E_config,verbose=False)
        E_flts = S_flts

        # _compute_classification_E(E,utterance.phns,E_flts,
        #                           phn_classification_dict,
        #                           linear_filters_cs,
        #                           phone_label,
        #                           bgd,
        #                           window_half_length,
        #                           data_file_idx
        #                           )

    return phn_classification_dict



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

def match_noise_to_file(noise,s,db):
    """
    Parameters
    ==========
    noise: numpy.ndarray[ndim=1]
        Noise to match to the file
    s: numpy.ndarray[ndim=1]
        Utterance to have the noise matched in length
    db: float
        Number of decibels between the two files
    """
    num_reps = len(s)/len(noise)+1
    repnoise = np.tile(noise,num_reps)
    repnoise = repnoise[:len(s)]
    noise_multiplier = np.linalg.norm(s)/np.linalg.norm(repnoise) * 1./(10.**(db/20.))
    return s + noise_multiplier*repnoise


def makeUtterance(
                 utterance_directory,
                 file_idx,
                 use_noise_file=None,
                 noise_db=0):
    s = np.load(utterance_directory+file_idx+'s.npy')
    if use_noise_file is not None:
        noise = np.load(use_noise_file)
        s = match_noise_to_file(noise,s,noise_db)

    return Utterance(
        utterance_directory=utterance_directory,
        file_idx=file_idx,
        s = s,
        phns = np.load(utterance_directory+file_idx+'phns.npy'),
        flts = np.load(utterance_directory+file_idx+'feature_label_transitions.npy'))




def makeTimeMap(old_start,old_end,new_start,new_end):
    return (np.arange(new_start,new_end) * float(old_end-old_start)/float(new_end-new_start) + .5).astype(int)


SyllableFeatures = collections.namedtuple("SyllableFeatures",
                                          "s S S_config E E_config offset phn_context assigned_phns"
                                          +" utt_path"
                                          +" file_idx"
                                          +" start_end")

def get_phn_context(start,end,phns,flts,offset=1,return_flts_context=False):
    phns_app = np.append(phns,'')
    prefix_phns = np.arange(phns_app.shape[0])[flts[:-1] <= start]
    if len(prefix_phns) > 0:
        start_idx = prefix_phns[-1]
    else:
        start_idx = 0
    suffix_phns = np.arange(phns_app.shape[0])[flts > end]
    if len(suffix_phns)> 0:
        end_idx = suffix_phns[0]
    else:
        end_idx = len(phns_app)
    if return_flts_context:
        return (np.hstack(
                (np.zeros((-min(start_idx-offset,0),),dtype=phns_app.dtype),
                 phns_app[start_idx:end_idx],
                 np.zeros(-min(end_idx-offset,0),dtype=phns_app.dtype))),
                np.hstack(
                ((-1)*np.ones((-min(start_idx-offset,0),),dtype=flts.dtype),
                 flts[start_idx:end_idx],
                 (-1)*np.ones(-min(end_idx-offset,0),dtype=flts.dtype))))

    else:
        return np.hstack(
            (np.zeros((-min(start_idx-offset,0),),dtype=phns_app.dtype),
             phns_app[start_idx-offset:end_idx+offset],
             np.zeros(-min(end_idx-offset,0),dtype=phns_app.dtype)))



def phns_syllable_matches(phns,syllable):
    syllable_len = len(syllable)
    return np.array([ phn_id
                      for phn_id in xrange(len(phns)-syllable_len+1)
                      if np.all(phns[phn_id:phn_id+syllable_len]==np.array(syllable))])

def get_example_over_interval(E,start_idx,end_idx,bgd=None):
    len_E = len(E)
    if len_E <end_idx:
        return np.vstack((E[start_idx:],
                          np.zeros(end_idx-len_E,E.shape[1:])))
    else:
        return E[start_idx:end_idx]


def get_example_with_offset(F,offset,start_idx,end_idx,default_val=0):
    if len(F.shape) > 1:
        if end_idx <= F.shape[0]:
            return np.vstack(
                (default_val * np.ones((-min(start_idx-offset,0),)+F.shape[1:],dtype=F.dtype),
                 F[start_idx:end_idx],
                 default_val * np.ones((-min(end_idx-offset,0),) + F.shape[1:],dtype=F.dtype)))
        else:
            return np.vstack(
                (default_val * np.ones((-min(start_idx-offset,0),)+F.shape[1:],dtype=F.dtype),
                 F[start_idx:end_idx],
                 default_val * np.ones((end_idx-F.shape[0],)+F.shape[1:],dtype=F.dtype),
                 default_val * np.ones((-min(end_idx-offset,0),) + F.shape[1:],dtype=F.dtype)))
    elif type(F[0]) == np.string_:
        return np.hstack(
            (np.zeros((-min(start_idx-offset,0),),dtype=F.dtype),
             F[start_idx:end_idx],
             np.zeros(-min(end_idx-offset,0),dtype=F.dtype)))
    else:
        return np.hstack(
            (default_val * np.ones((-min(start_idx-offset,0),),dtype=F.dtype),
             F[start_idx:end_idx],
             default_val * np.ones(-min(end_idx-offset,0),dtype=F.dtype)))

def get_syllable_features(utterance_directory,data_idx,syllable,
                          S_config=None,E_config=None,offset = None,E_verbose=False,avg_bgd=None,
                          waveform_offset=0,
                          phn_mapping = None,
                          assigned_phns = None,
                          P_config=None,
                          verbose=False,
                          mel_smoothing_kernel=-1,
                          avg_bgd_spec=None):
    """
    Expects a list of (name, parameters) tuples
    names are:
       waveform
       spectrogram
       edge_map
       waliji

    Parameters:
    ===========
    utterances_path: str
       the TIMIT examples are assumed to be stored in this directory.
    waveform_offset:
        The offset is assumed to be a positive integer where we offset
        the waveform collection by waveform_offset * S_config.num_window_samples

    """
    if verbose:
        print "%s%s" % (utterance_directory,data_idx)
    utterance = makeUtterance(utterance_directory,data_idx)
    sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
    # get the spectrogram
    if S_config is not None:
        S = get_spectrogram(utterance.s,S_config,
                            mel_smoothing_kernel=mel_smoothing_kernel)


        if avg_bgd_spec is not None:
            avg_bgd_spec.add_frames(S,time_axis=0)
        S_flts = (sflts * S.shape[0] /float(sflts[-1]) + .5).astype(int)
        if E_config is not None:
            E = get_edge_features(S.T,E_config,verbose=E_verbose)
            # both are the same
            E_flts = S_flts

            if P_config is not None:
                P = get_part_features(E,P_config)
                P_flts=E_flts.copy()
                P_flts[-1] = len(P)
                if avg_bgd is not None:
                    avg_bgd.add_frames(P,time_axis=0)
            elif avg_bgd is not None:
                avg_bgd.add_frames(E,time_axis=0)
                P=None
            else:
                P=None

        else:
            E = None
            P = None
    else:
        S = None
        E = None
        P=None
    # we then get the example phones removed from the signals
    use_phns = utterance.phns.copy()
    if phn_mapping is not None:
        use_phns[:] = np.array([phn_mapping[p] for p in use_phns])
    syllable_starts = phns_syllable_matches(use_phns,syllable)
    syllable_length = len(syllable)



    if (waveform_offset is None or waveform_offset == 0) and (S is not None and P is not None):
        return [ SyllableFeatures(
                s = (utterance.s)[sflts[syllable_start]:sflts[syllable_start+syllable_length]],
                S = get_example_with_offset(S,offset,S_flts[syllable_start],S_flts[syllable_start+syllable_length]),
                S_config = S_config,
                E = get_example_with_offset(P,offset,P_flts[syllable_start],P_flts[syllable_start+syllable_length]),
                E_config = {'E':E_config, 'P':P_config},
                offset = offset,
                phn_context = get_phn_context(syllable_start,
                                              syllable_start+syllable_length,
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns = syllable,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=(syllable_start,syllable_start+syllable_length))
                 for syllable_start in syllable_starts]
    elif (waveform_offset > 0) and (S is not None and P is not None):
        return [ SyllableFeatures(
                s = get_example_with_offset(utterance.s,
                                            waveform_offset,
                                            sflts[syllable_start],
                                            sflts[syllable_start+syllable_length],
                                            default_val=0),
                S = get_example_with_offset(S,offset,S_flts[syllable_start],S_flts[syllable_start+syllable_length]),
                S_config = S_config,
                E = get_example_with_offset(P,offset,P_flts[syllable_start],P_flts[syllable_start+syllable_length]),
                E_config = {'E':E_config, 'P':P_config},
                offset = offset,
                phn_context = get_phn_context(syllable_start,
                                              syllable_start+syllable_length,
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns=syllable,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=(syllable_start,syllable_start+syllable_length))
                 for syllable_start in syllable_starts]

    elif (waveform_offset is None or waveform_offset == 0) and (S is not None and E is not None):
        return [ SyllableFeatures(
                s = (utterance.s)[sflts[syllable_start]:sflts[syllable_start+syllable_length]],
                S = get_example_with_offset(S,offset,S_flts[syllable_start],S_flts[syllable_start+syllable_length]),
                S_config = S_config,
                E = get_example_with_offset(E,offset,E_flts[syllable_start],E_flts[syllable_start+syllable_length]),
                E_config = E_config,
                offset = offset,
                phn_context = get_phn_context(syllable_start,
                                              syllable_start+syllable_length,
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns = syllable,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=(syllable_start,syllable_start+syllable_length))
                 for syllable_start in syllable_starts]
    elif (waveform_offset > 0) and (S is not None and E is not None):
        return [ SyllableFeatures(
                s = get_example_with_offset(utterance.s,
                                            waveform_offset,
                                            sflts[syllable_start],
                                            sflts[syllable_start+syllable_length],
                                            default_val=0),
                S = get_example_with_offset(S,offset,S_flts[syllable_start],S_flts[syllable_start+syllable_length]),
                S_config = S_config,
                E = get_example_with_offset(E,offset,E_flts[syllable_start],E_flts[syllable_start+syllable_length]),
                E_config = E_config,
                offset = offset,
                phn_context = get_phn_context(syllable_start,
                                              syllable_start+syllable_length,
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns=syllable,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=(syllable_start,syllable_start+syllable_length))
                 for syllable_start in syllable_starts]
    else:
        return None


def get_syllable_features_cluster(utterance_directory,data_idx,cluster_list,
                          S_config=None,E_config=None,P_config=None,offset =0,E_verbose=False,avg_bgd=None,
                          waveform_offset=0,
                                  assigned_phns = None,verbose=False,detect_length=None):
    """
    Expects a list of (name, parameters) tuples
    names are:
       waveform
       spectrogram
       edge_map
       waliji

    Parameters:
    ===========
    utterances_path: str
       the TIMIT examples are assumed to be stored in this directory.
    waveform_offset:
        The offset is assumed to be a positive integer where we offset
        the waveform collection by waveform_offset * S_config.num_window_samples

    Output:
    =======
    """
    utterance = makeUtterance(utterance_directory,data_idx)
    sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
    # get the spectrogram

    if S_config is not None:
        S = get_spectrogram(utterance.s,S_config)
        cluster_end = max( c[1] for c in cluster_list)
        S_flts = (sflts * S.shape[0] /float(sflts[-1]) + .5).astype(int)
        if E_config is not None:
            E = get_edge_features(S.T,E_config,verbose=E_verbose)
            if avg_bgd is not None:
                avg_bgd.add_frames(E,time_axis=0)
            # both are the same
            E_flts = S_flts

            if detect_length is not None:
                E_row_shape = E[0].shape
                E = np.vstack((E,np.zeros((detect_length,)+E_row_shape)))
                S = np.vstack((S,np.tile(S[-5:].mean(0),(detect_length,1))))


            if P_config is not None:
                P = get_part_features(E,P_config)
                P_flts=E_flts.copy()
                P_flts[-1] = len(P)
                if avg_bgd is not None:
                    avg_bgd.add_frames(P,time_axis=0)
            elif avg_bgd is not None:
                avg_bgd.add_frames(E,time_axis=0)
                P=None
            else:
                P=None

        else:
            E = None
            P=None

    else:
        S = None
        E = None
        P=None

    # we then get the example phones removed from the signals



    s_cluster_list = tuple(
        tuple( int(v * sflts[-1]/float(S.shape[0]) + .5)
               for v in cluster)
        for cluster in cluster_list
        )

    if (waveform_offset is None or waveform_offset == 0) and (S is not None and P is not None):
        return tuple( SyllableFeatures(
                s = (utterance.s)[s_cluster[0]:s_cluster[1]],
                S = S[cluster[0]:cluster[1]],
                S_config = S_config,
                E = P[cluster[0]:cluster[1]],
                E_config = {'E':E_config,'P':P_config},
                offset = 0,
                phn_context =get_phn_context(cluster[0],
                                              cluster[1],
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns = assigned_phns,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=cluster)
                 for s_cluster,cluster in itertools.izip(s_cluster_list,cluster_list))
    elif (waveform_offset > 0) and (S is not None and P is not None):
        for s_cluster,cluster in itertools.izip(s_cluster_list,cluster_list):
            if cluster[1] > P.shape[0]:
                print "one cluster is too big: %d" % (cluster[1]-P.shape[0])


        return tuple( SyllableFeatures(
                s = get_example_with_offset(utterance.s,
                                            waveform_offset,
                                            s_cluster[0],
                                            s_cluster[1],
                                            default_val=0),
                S = get_example_with_offset(S,offset,cluster[0],cluster[1]),
                S_config = S_config,
                E = get_example_with_offset(P,offset,cluster[0],cluster[1]),
                E_config = {'E':E_config,'P':P_config},
                offset = offset,
                phn_context = get_phn_context(cluster[0],
                                              cluster[1],
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns = assigned_phns,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=cluster)
                 for s_cluster,cluster in itertools.izip(s_cluster_list,cluster_list))
    elif (waveform_offset is None or waveform_offset == 0) and (S is not None and E is not None):
        return tuple( SyllableFeatures(
                s = (utterance.s)[s_cluster[0]:s_cluster[1]],
                S = S[cluster[0]:cluster[1]],
                S_config = S_config,
                E = E[cluster[0]:cluster[1]],
                E_config = E_config,
                offset = 0,
                phn_context =get_phn_context(cluster[0],
                                              cluster[1],
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns = assigned_phns,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=cluster)
                 for s_cluster,cluster in itertools.izip(s_cluster_list,cluster_list))
    elif (waveform_offset > 0) and (S is not None and E is not None):
        if verbose:
            print "waveform_offset=%d, S is not none, E is not None" % waveform_offset
            for cluster_id,cluster in enumerate(cluster_list):
                print "cluster_id=%d, cluster=(%d,%d)\tlength=%d\tlen E=%d" %(cluster_id, cluster[0],cluster[1],len(E[cluster[0]:cluster[1]]),len(E))
        return tuple( SyllableFeatures(
                s = get_example_with_offset(utterance.s,
                                            waveform_offset,
                                            s_cluster[0],
                                            s_cluster[1],
                                            default_val=0),
                S = S[cluster[0]:cluster[1]],
                S_config = S_config,
                E = E[cluster[0]:cluster[1]],
                E_config = E_config,
                offset = offset,
                phn_context = get_phn_context(cluster[0],
                                              cluster[1],
                                              utterance.phns,
                                              utterance.flts),
                assigned_phns = assigned_phns,
                utt_path=utterance_directory,
                file_idx=data_idx,
                start_end=cluster)
                 for s_cluster,cluster in itertools.izip(s_cluster_list,cluster_list))
    else:
        return None




def get_syllable_features_directory(utterances_path,file_indices,syllable,
                                    S_config=None,E_config=None,offset=None,
                                    E_verbose=False,return_avg_bgd=True,waveform_offset=0,
                                  phn_mapping=None,P_config=None,verbose=False,
                                    mel_smoothing_kernel=-1,do_parallel=False,
                                    do_avg_bgd_spec=False):
    """
    Parameters:
    ===========
    utterances_path: str
       the TIMIT examples are assumed to be stored in this directory.
    waveform_offset:

        The offset is assumed to be a positive integer where we offset
        the waveform collection by waveform_offset * S_config.num_window_samples
    P_Config: None or PartsParameters
        Either None (in which case we don't do parts processing) or PartsParameters
    """
    avg_bgd = AverageBackground()

    avg_bgd_spec=AverageBackground()
    return_tuple = tuple(
        get_syllable_features(utterances_path,data_idx,syllable,
                              S_config=S_config,E_config=E_config,offset = offset,
                              E_verbose=E_verbose,avg_bgd=avg_bgd,
                              waveform_offset=waveform_offset,
                              phn_mapping=phn_mapping,
                              P_config=P_config,
                              verbose=verbose,
                              mel_smoothing_kernel=mel_smoothing_kernel,
                              avg_bgd_spec=avg_bgd_spec)
        for data_idx in file_indices)
    if return_avg_bgd and avg_bgd_spec:
        return return_tuple, avg_bgd, avg_bgd_spec
    elif return_avg_bgd:
        return return_tuple, avg_bgd
    else:
        return return_tuple

def recover_example_map(syllable_features):
    return np.array(reduce(lambda x,y : x + y,
                           [[i] * len(e)
                            for i,e in enumerate(syllable_features)]))

def recover_edgemaps(syllable_features,example_mat,bgd=None):
    max_length = max(
        max((0,) + tuple(s.E.shape[0] for s in e))
        for e in syllable_features)
    for e in syllable_features:
        if len(e) > 0: break
    E_shape = e[0].E.shape[1:]
    Es = np.zeros((len(example_mat),max_length)+E_shape,dtype=np.uint8)
    lengths = np.zeros(len(example_mat))
    jdx=0
    for idx,i in enumerate(example_mat):
        if idx > 0:
            if i == example_mat[idx-1]:
                jdx += 1
            else:
                jdx = 0

            lengths[idx] = len(syllable_features[i][jdx].E)
            Es[idx][:lengths[idx]] = syllable_features[i][jdx].E.astype(np.uint8)[:lengths[idx]]
            if bgd is not None and lengths[idx] <max_length:
                Es[idx][lengths[idx]:] = (np.random.rand(
                    *((max_length - lengths[idx],)+E_shape))
                                          <= np.tile(bgd,
                                                    (max_length-lengths[idx],
                                                     1,1))).astype(np.uint8)
        else:
            lengths[idx] = len(syllable_features[i][jdx].E)
            Es[idx][:lengths[idx]] = syllable_features[i][jdx].E.astype(np.uint8)[:lengths[idx]]
            if bgd is not None and lengths[idx] <max_length:
                Es[idx][lengths[idx]:] = (np.random.rand(
                    *((max_length - lengths[idx],)+E_shape))
                                          <= np.tile(bgd,
                                                    (max_length-lengths[idx],
                                                     1,1))).astype(np.uint8)

    return lengths, Es

def recover_specs(syllable_features,example_mat,
                  bgd=None,bgd_std=None):
    max_length = max(
        max((0,) + tuple(s.S.shape[0] for s in e))
        for e in syllable_features)
    for e in syllable_features:
        if len(e) > 0: break
    try:
        S_shape = e[0].S.shape[1:]
    except:
        import pdb; pdb.set_trace()
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
            if bgd is not None and lengths[idx] < max_length:
                Ss[idx][lengths[idx]:] = (
                    np.random.randn(*((max_length-lengths[idx],)
                                    + S_shape)) * bgd_std + np.tile(bgd,
                                                 (max_length-lengths[idx],1)))
        else:
            lengths[idx] = len(syllable_features[i][jdx].S)
            Ss[idx][:lengths[idx]] = syllable_features[i][jdx].S
            if bgd is not None and lengths[idx] < max_length:

                Ss[idx][lengths[idx]:] = (
                    np.random.randn(*((max_length-lengths[idx],)
                                    + S_shape)) * bgd_std + np.tile(bgd,
                                                 (max_length-lengths[idx],1)))

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


def recover_assigned_phns(syllable_features,example_mat):
    assigned_phns = np.empty(len(example_mat),dtype=object)
    phn_contexts = np.empty(len(example_mat),dtype=object)
    utt_paths = np.empty(len(example_mat),dtype=object)
    file_indices = np.empty(len(example_mat),dtype=object)
    start_ends = np.empty((len(example_mat),2),dtype=int)
    jdx=0
    for idx,i in enumerate(example_mat):
        if idx > 0:
            if i == example_mat[idx-1]:
                jdx += 1
            else:
                jdx = 0

            assigned_phns[idx] = syllable_features[i][jdx].assigned_phns
            phn_contexts[idx] = syllable_features[i][jdx].phn_context
            utt_paths[idx] = syllable_features[i][jdx].utt_path
            file_indices[idx] = syllable_features[i][jdx].file_idx
            start_ends[idx,:] = np.array( syllable_features[i][jdx].start_end)
        else:
            assigned_phns[idx] = syllable_features[i][jdx].assigned_phns
            phn_contexts[idx] = syllable_features[i][jdx].phn_context
            utt_paths[idx] = syllable_features[i][jdx].utt_path
            file_indices[idx] = syllable_features[i][jdx].file_idx
            start_ends[idx,:] = np.array( syllable_features[i][jdx].start_end)
    return assigned_phns,phn_contexts, utt_paths, file_indices, start_ends



SpectrogramParameters = collections.namedtuple("SpectrogramParameters",
                                               ("sample_rate"
                                                +" num_window_samples"
                                                +" num_window_step_samples"
                                                +" fft_length"
                                                +" kernel_length"
                                                +" freq_cutoff"
                                                +" use_mel"
                                                +" mel_smoothing_kernel"
                                                +" use_dss"
                                                +" do_mfccs"
                                                +" nbands"
                                                +" num_ceps"
                                                +" liftering"
                                                +" include_energy"
                                                +" include_deltas"
                                                +" include_double_deltas"
                                                +" delta_window"
                                                +" no_use_dpss"
                                                +" do_freq_smoothing"))

def makeSpectrogramParameters(sample_rate=16000,
                              num_window_samples=320,
                              num_window_step_samples=80,
                              fft_length=512,
                              kernel_length=7,
                              freq_cutoff=3000,
                              use_mel=False,
                              mel_smoothing_kernel=-1,
                              use_dss=True,
                              do_mfccs=False,
                              nbands=40,
                              num_ceps=13,
                              liftering=.6,
                              include_energy=False,
                              include_deltas=False,
                              include_double_deltas=False,
                              delta_window=9,
                              no_use_dpss=False,
                              do_freq_smoothing=True):
    """
    wrapper Function so that named tuple can have
    optional arguments

    Parameters:
    ===========
    use_dss
    """
    return SpectrogramParameters(sample_rate=sample_rate,
                                 num_window_samples=num_window_samples,
                                 num_window_step_samples=num_window_step_samples,
                                 fft_length=fft_length,
                                 kernel_length=kernel_length,
                                 freq_cutoff=freq_cutoff,
                                 use_mel=use_mel,
                                 mel_smoothing_kernel=mel_smoothing_kernel,
                                 use_dss=use_dss,
                                 do_mfccs=do_mfccs,
                                 nbands=nbands,
                                 num_ceps=num_ceps,
                                 liftering=liftering,
                                 include_energy=include_energy,
                                 include_deltas=include_deltas,
                                 include_double_deltas=include_double_deltas,
                                 delta_window=delta_window,
                                 no_use_dpss=no_use_dpss,
                                 do_freq_smoothing=do_freq_smoothing)

def get_spectrogram(waveform,spectrogram_parameters,mel_smoothing_kernel=-1):
    if spectrogram_parameters.use_mel:
        return esp.get_mel_spec(waveform,
                            spectrogram_parameters.sample_rate,
                        spectrogram_parameters.num_window_samples,
                        spectrogram_parameters.num_window_step_samples,
                        spectrogram_parameters.fft_length,
                                nbands=spectrogram_parameters.nbands,
                                mel_smoothing_kernel=spectrogram_parameters.mel_smoothing_kernel,
                                num_ceps=spectrogram_parameters.num_ceps,
                                lift=spectrogram_parameters.liftering,
                                include_energy=spectrogram_parameters.include_energy,
                                no_use_dpss=spectrogram_parameters.no_use_dpss
                                ).T
    elif spectrogram_parameters.do_mfccs:
        return esp.get_melfcc(waveform,
                            spectrogram_parameters.sample_rate,
                        spectrogram_parameters.num_window_samples,
                        spectrogram_parameters.num_window_step_samples,
                        spectrogram_parameters.fft_length,
                                nbands=spectrogram_parameters.nbands,
                              num_ceps=spectrogram_parameters.num_ceps,
                              lift=spectrogram_parameters.liftering,
                              include_energy=spectrogram_parameters.include_energy,
                              include_deltas=spectrogram_parameters.include_deltas,
                              include_double_deltas=spectrogram_parameters.include_double_deltas,
                              delta_window=spectrogram_parameters.delta_window,
                              no_use_dpss=spectrogram_parameters.no_use_dpss
                                ).T
    else:
        return esp.get_spectrogram_features(waveform,
                                     spectrogram_parameters.sample_rate,
                                     spectrogram_parameters.num_window_samples,
                                     spectrogram_parameters.num_window_step_samples,
                                     spectrogram_parameters.fft_length,
                                     spectrogram_parameters.freq_cutoff,
                                     spectrogram_parameters.kernel_length,
                                            no_use_dpss=spectrogram_parameters.no_use_dpss,
                                            do_freq_smoothing=spectrogram_parameters.do_freq_smoothing
                                 ).T

EdgemapParameters = collections.namedtuple("EdgemapParameters",
                                           ("block_length"
                                            +" spread_length"
                                            +" threshold"
                                            +" magnitude_features"
                                            +" magnitude_block_length"
                                            +" magnitude_and_edge_features"))

def makeEdgemapParameters(block_length,
                          spread_length,
                          threshold,
                          magnitude_features=False,
                          magnitude_block_length=0,
                          magnitude_and_edge_features=False):
    """
    Wrapper function to construct EdgemapParameters
    named tuple

    Parameters:
    ===========
    block_length: int
        Number of frames to use for estimating the cutoff percentile
    spread_length: int
        radius for pixel spreading
    threshold: float
        percentile to threshold values at
    magnitude_features: bool
        whether to just use the magnitudes of the feature values rather
        than computing the edgemaps
    """
    if magnitude_block_length == 0:
        magnitude_block_length = block_length
    emp = EdgemapParameters(block_length=block_length,
                            spread_length=spread_length,
                            threshold=threshold,
                            magnitude_features=magnitude_features,
                            magnitude_block_length=magnitude_block_length,
                            magnitude_and_edge_features=magnitude_and_edge_features)
    return emp

PartsParameters = collections.namedtuple("PartsParameters",
                                           ("use_parts"
                                            +" parts_path"
                                            +" bernsteinEdgeThreshold"
                                            +" logParts"
                                            +" logInvParts"
                                            +" spreadRadiusX"
                                            +" spreadRadiusY"
                                            +" numParts"
                                            +" partGraph"
                                            +" parts_S"
                                            +" parts_mask"))

def makePartsParameters(use_parts,
                        parts_path,
                        bernsteinEdgeThreshold,
                        logParts,
                        logInvParts,
                        spreadRadiusX,
                        spreadRadiusY,
                        numParts,
                        partGraph=None,
                        parts_S=None,
                        parts_mask=None):
    if parts_mask is None:
        parts_mask = np.ones(logParts.shape[1:-1],dtype=np.uint8)

    pp2= PartsParameters(use_parts=use_parts,
                           parts_path=parts_path,
                           bernsteinEdgeThreshold=bernsteinEdgeThreshold,
                           logParts=logParts,
                           logInvParts=logInvParts,
                           spreadRadiusX=spreadRadiusX,
                           spreadRadiusY=spreadRadiusY,
                           numParts=numParts,
                         partGraph=partGraph,
                         parts_S=parts_S,
                         parts_mask=parts_mask)


    return pp2


def get_edge_features(S,parameters,verbose=False):
    if parameters.magnitude_features:
        E = esp.magnitude_features(S,
                           parameters.block_length,
                           parameters.spread_length,
                           parameters.threshold)
        return E.reshape(E.shape[0],E.shape[1],1)
    elif parameters.magnitude_and_edge_features:
        E2 = esp.magnitude_features(S,
                           parameters.block_length,
                           parameters.spread_length,
                           parameters.threshold)
        E, edge_feature_row_breaks,\
            edge_orientations = esp._edge_map_no_threshold(S)
        esp._edge_map_threshold_segments(E,
                                     parameters.block_length,
                                     parameters.spread_length,
                                     threshold=parameters.threshold,
                                     edge_orientations = edge_orientations,
                                     edge_feature_row_breaks = edge_feature_row_breaks,
                                     verbose=verbose)
        E = reorg_part_for_fast_filtering(E)
        return np.dstack((E,E2))
    else:
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

def get_part_features(E,parameters,verbose=False):
    out = cp.code_parts(E.astype(np.uint8),
                        parameters.logParts,parameters.logInvParts,parameters.bernsteinEdgeThreshold)
    max_responses = np.argmax(out,-1)
    spread_responses = cp.spread_patches(max_responses,
                                         parameters.spreadRadiusX,
                                         parameters.spreadRadiusY,
                                         parameters.numParts)
    if parameters.partGraph is not None:
        cp.spread_graph_patches(spread_responses,
                          parameters.partGraph)
    return spread_responses

