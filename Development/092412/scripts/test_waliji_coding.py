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

# going to check the code inside the part estimation data to make sure
# that I understand what coordinates in these parts represents what
# since I'm going to change the parts so that they are time x frequency x part_id
# file to check is in
# Experiments/090712/scripts/check_part_plausibility.py
# it is determined by the way the feature extraction happens

# need to test what is going on

import template_speech_rec.extract_local_features as elf

E = np.arange(200).reshape(10,20)

patches = elf._extract_block_local_features_tied(E,3,3)

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


# now we need to test whether that works/makes sense
E = np.vstack(tuple(
        (np.arange(24).reshape(4,6) % 6) + i*6
        for i in xrange(8)))

E_reorg = reorg_part_for_fast_filtering(E)

# load in a speech file

reorg_parts = reorg_parts_for_fast_filtering(parts)

# test that the part filtering works correctly
# generate a test example where the parts occur 1 by 1 across rows and make sure
# that we recover the right parts

np.random.seed(0)
E_vals = np.random.rand(10*5,50*5,8)
E = np.zeros((10*5,50*5,8))
for i in xrange(10):
    for j in xrange(50):
        E[i*5:(i+1)*5,
          j*5:(j+1)*5] = (E_vals[i*5:(i+1)*5,
                                 j*5:(j+1)*5] < reorg_parts[j])


log_part_blocks = np.log(reorg_parts)
log_invpart_blocks = np.log(1-reorg_parts)

def get_part_scores(E,log_part_blocks,log_invpart_blocks):
    part_block_shape = log_part_blocks[0].shape
    t_start = part_block_shape[0]/2
    t_end = t_start + E.shape[0] - part_block_shape[0] + 1
    f_start = part_block_shape[1]/2
    f_end = f_start + E.shape[1] - part_block_shape[1] + 1
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


def code_parts(E,log_part_blocks,log_invpart_blocks):
    """
    Assumes that E has been reorganized to have
    dimension 0 be the time axis
    dimension 1 be the frequency axis
    dimension 2 be the edge type axis
    
    """
    return np.argmax(get_part_scores(E,log_part_blocks,log_invpart_blocks),0)



E_coded = code_parts(E,log_part_blocks,log_invpart_blocks)
# this was tested by observing that 
for i in xrange(0,
                E_coded.shape[0],
                5):
    assert np.all(E_coded[i][::5] == np.arange(50))





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


E_collapsed = collapse_to_grid(E_coded,5,5,50)


#
#
# now all we need to do  is get examples
#
#
# we are going to construct a training and validatin set
# these are 2/3rds of the data and 1/3 of the training data
#
# 
#


#
# training portion of the data
#

