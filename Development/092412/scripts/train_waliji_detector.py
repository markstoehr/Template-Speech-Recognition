import numpy as np
from scipy import ndimage

root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle
sys.path.append(root_path)
from scipy.ndimage.filters import generic_filter


# set arbitrary initial parameters
lower_cutoff = 30
num_parts = 50

# retrieve the parts
parts = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))


import template_speech_rec.extract_local_features as elf

#
# create function to adjust parts into a useable form for filtering
#
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
            
def reorg_parts_for_fast_filtering(parts,feature_types=8):
    return np.array([
            reorg_part_for_fast_filtering(part,feature_types=feature_types)
            for part in parts])


reorg_parts = reorg_parts_for_fast_filtering(parts)

log_part_blocks = np.log(reorg_parts)
log_invpart_blocks = np.log(1-reorg_parts)

def get_part_scores(E,log_part_blocks,log_invpart_blocks,
                    frequency_mode='valid',time_mode='same'):
    part_block_shape = log_part_blocks[0].shape
    if time_mode=='valid':
        t_start = part_block_shape[0]/2
        t_end = t_start + E.shape[0] - part_block_shape[0] + 1
    else: # feature_mode == 'same'
        # TODO: check whether I should make the modes available
        #       type-safe
        t_start = 0
        t_end = E.shape[0]
    # check the frequency parameters
    if frequency_mode=='valid':
        f_start = part_block_shape[1]/2
        f_end = f_start + E.shape[1] - part_block_shape[1] + 1
    else: # frequency_mode == 'same'
        f_start = 0
        f_end = E.shape[1]
    e_pos = part_block_shape[2]/2
    return np.array([
            (ndimage.correlate(E,log_part_block)[t_start:t_end,
                                                 f_start:f_end,
                                                 e_pos]
             +ndimage.correlate(1-E,log_invpart_block)[t_start:t_end,
                                                       f_start:f_end,
                                                       e_pos])
            for log_part_block,log_invpart_block in zip(log_part_blocks,
                                                        log_invpart_blocks)])


def code_parts(E,log_part_blocks,log_invpart_blocks,edge_count_lower_bound=30):
    """
    Assumes that E has been reorganized to have
    dimension 0 be the time axis
    dimension 1 be the frequency axis
    dimension 2 be the edge type axis
    
    Parameters:
    ===========
    log_part_blocks: ndarray[ndim=4,dtype=float]        
        Dimension 0 is the part index, dimension 1 is a time index
        dimension 2 is a frequency index, dimension 3 is an edge type
        index.  These are log probabilities over the occurance of an
        edge of the particular type at those relative time and
        frequency locations

    log_invpart_blocks: ndarray[ndim=4,dtype=float]
        Same information as the array in log_part_blocks, assumed to be
        precomputed for time saving this is equal to:
        
                   np.log( 1. - np.exp( log_part_blocks))
                   
        hence this allows for fast bernoulli model computation.

    Output:
    =======
    Part_Index_Map: ndarray[ndim=2,dtype=int]
        This has only two dimensions and those dimensions are equal
        to the first two dimensions of E. The values of these entries
        are the part indices.  

    
    """
    part_scores = get_part_scores(
        E,
        np.vstack((np.ones((1,)+log_part_blocks.shape[1:]),
                   log_part_blocks)),
        np.vstack((np.zeros((1,)+log_part_blocks.shape[1:]),
                   log_invpart_blocks)),
        frequency_mode='valid',time_mode='same')
    # get upper bound on the part scores
    part_score_upper_bound_strict = E.max()+1.
    # find the significant counts
    part_scores[0][part_scores[0] < edge_count_lower_bound] = part_score_upper_bound_strict
    return np.argmax(
        part_scores,0)




def collapse_to_grid(E_coded,grid_time,grid_frequency,num_codes,
                     do_grid_subsampling =True):
    full_grid = np.dstack(tuple(
            generic_filter(E_coded,lambda x: np.any(x==i).astype(np.uint8),
                           size = (grid_time,
                                   grid_frequency))
            for i in xrange(num_codes)))
    if do_grid_subsampling:
        return full_grid[::grid_time,::grid_frequency]
    else:
        return full_grid

train_ids = open(data_path+'train_ids.txt','r').read().strip().split('\n')

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
import template_speech_rec.template_experiments as t_exp

from collections import defaultdict

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

spread_length=3
abst_threshold=abst_threshold
fft_length=512
num_window_step_samples=80
freq_cutoff=3000
sample_rate=16000
num_window_samples=320
kernel_length=7
offset=3

syllables = np.array([['aa','r'],['p','aa'],['t','aa'],['k','aa'],['b','aa'],['d','aa'],['g','aa']])
example_E_dict = defaultdict(list)
example_S_dict = defaultdict(list)
example_locs = {}

log_part_blocks = np.log(reorg_parts)
log_invpart_blocks = np.log(1-reorg_parts)


def phns_syllable_matches(phns,syllable):
    syllable_len = len(syllable)
    return [ phn_id
            for phn_id in xrange(len(phns)-syllable_len+1)
            if np.all(phns[phn_id:phn_id+syllable_len]==syllable)]



avg_bgd_obj = t_exp.AverageBackground()

aar_examples = []
aar = np.array(['aa','r'])

for i,idx in enumerate(train_ids):
    print i
    s = np.load(data_path+'Train/'+idx+'s.npy')
    phns = np.load(data_path+'Train/'+idx+'phns.npy')
    # we divide by 5 since we coarsen in the time domain
    flts = np.load(data_path+'Train/'+idx+'feature_label_transitions.npy')
    aar_locs = phns_syllable_matches(phns,aar)
    if len(aar_locs) == 0: continue
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
    F = code_parts(E,log_part_blocks,log_invpart_blocks,
                   edge_count_lower_bound=30)
    F = collapse_to_grid(F,log_part_blocks.shape[1],
                                   log_part_blocks.shape[2],
                                   log_part_blocks.shape[0])
    print E.shape
    E_2d = np.hstack(tuple(
            E[:,:,i]
            for i in xrange(E.shape[2]))).T
    avg_bgd_obj.add_frames(E_2d)
    syllable_locs = [phns_syllable_matches(phns,syllable) for syllable in syllables]
    for syllable_id, syllable_loc_list in enumerate(syllable_locs):
        syllable = tuple(syllables[syllable_id])
        if syllable[0]=='aa' and syllable[1] == 'r':
            aar_examples.extend([
                    E[max(flts[loc]/5,0):min((flts[loc+len(syllable)]+1)/5,E.shape[0])]
                    for loc in syllable_loc_list])
        example_E_dict[syllable].extend([
            E[max(flts[loc]/5,0):min((1+flts[loc+len(syllable)])/5,E.shape[0])]
            for loc in syllable_loc_list])


    


