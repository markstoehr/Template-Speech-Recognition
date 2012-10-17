import numpy as np
from scipy import ndimage

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle,re,itertools
sys.path.append(root_path)

import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
import template_speech_rec.code_parts as cp
import template_speech_rec.spread_waliji_patches as swp

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
                      if np.all(phns[phn_id:phn_id+syllable_len]==syllable)])

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
    phn_matches = phns_syllable_matches(phns,syllable)
    if phn_matches.shape[0] ==0:
        return None, None
    syllable_starts = flts[ phn_matches]
    syllable_ends = flts[phn_matches + len(syllable)]
    syllable_starts = map_array_to_coarse_coordinates(syllable_starts,log_part_blocks.shape[1])
    syllable_ends = np.clip(map_array_to_coarse_coordinates(syllable_ends,
                                                            log_part_blocks.shape[1]),
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
                           kernel_length=7):
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
    E = reorg_part_for_fast_filtering(E)
    F = cp.code_parts_fast(E.astype(np.uint8),log_part_blocks,log_invpart_blocks,10)
    F = np.argmax(F,2)
    # the amount of spreading to do is governed by the size of the part features
    F = swp.spread_waliji_patches(F,
                                  log_part_blocks.shape[1],
                                  log_part_blocks.shape[2],
                                  log_part_blocks.shape[0])
    return collapse_to_grid(F,log_part_blocks.shape[1],
                            log_part_blocks.shape[2])




def _get_syllable_examples_background_files(train_data_path,
                                            data_file_idx,
                                            avg_bgd,
                                            syllable,
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
                                            kernel_length=7
                                            ):
    """
    Perform main signal processin
    """
    s = np.load(data_path+'Train/'+data_file_idx+'s.npy')
    phns = np.load(data_path+'Train/'+data_file_idx+'phns.npy')
    # we divide by 5 since we coarsen in the time domain
    flts = np.load(data_path+'Train/'+data_file_idx+'feature_label_transitions.npy')
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
                               kernel_length=kernel_length,
background_length=3)
    print "F has been estimated and it has shape ", str(F.shape)
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
        backgrounds.extend([
            F[max(0,s-background_length):s].mean(0)
            for s in example_starts])


def get_syllable_examples_backgrounds_files(train_data_path,
                                            data_files_indices,
                                            syllable,
                                            log_part_blocks,
                                            log_invpart_blocks,
                                            num_examples=-1,
                                            verbose=False):
    avg_bgd = AverageBackground()
    syllable_examples = []
    if num_examples == -1:
        num_examples = len(data_files_indices)
    backgrounds = []
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
                                                 log_invpart_blocks)
    return avg_bgd, syllable_examples, backgrounds



def get_training_examples(syllable,train_data_path):
    data_files_indices = get_data_files_indices(train_data_path)



