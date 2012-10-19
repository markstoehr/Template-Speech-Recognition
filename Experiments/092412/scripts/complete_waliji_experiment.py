# first we load in things
# to make this basic bit work
import numpy as np

root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle
sys.path.append(root_path)


##
# We assume that parts have already been computed
# we are not interested in comparing different part architectures at the moment
#

# load in the parts that we use for coding

lower_cutoff=10
num_parts = 50
# retrieve the parts
parts = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))


# perform basic transformation so its easy to use
# convert to a smaller type for our cython functions
import template_speech_rec.get_train_data as gtrd

log_part_blocks, log_invpart_blocks = gtrd.reorg_parts_for_fast_filtering(parts)
log_part_blocks = log_part_blocks.astype(np.float32)
log_invpart_blocks = log_invpart_blocks.astype(np.float32)


#
#
# now we get the examples that we are wanting
#

syllables = (('p','aa'),
             ('p','iy'),
             ('b','iy'),
             ('sh','aa'),
             ('sh','iy'),
             ('f','aa'),
             ('s','aa'),
             ('s','iy'),
             ('m','aa'),
             ('m','iy'),
             ('l','aa'),
             ('l','iy'),)
             

train_data_path = root_path+'Data/Train/'

file_indices = gtrd.get_data_files_indices(train_data_path)

avg_bgd, syllable_examples, backgrounds = gtrd.get_syllables_examples_backgrounds_files(train_data_path,
                                            file_indices,
                                            syllables,
                                            log_part_blocks,
                                            log_invpart_blocks,
                                            num_examples=-1,
                                            verbose=True)
