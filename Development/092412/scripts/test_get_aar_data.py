import numpy as np
from scipy import ndimage

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle
sys.path.append(root_path)




lower_cutoff=10
num_parts = 50
# retrieve the parts
parts = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))



log_part_blocks, log_invpart_blocks = gad.reorg_parts_for_fast_filtering(parts)



train_data_path = root_path+'Data/Train/'

file_indices = gad.get_data_files_indices(train_data_path)


s = np.load(train_data_path + file_indices[0]+'s.npy')
phns = np.load(train_data_path + file_indices[0]+'phns.npy')
flts = np.load(train_data_path + file_indices[0]+'feature_label_transitions.npy')

gad.phns_syllable_matches(phns,np.array(['dcl','d']))

F = gad.get_waliji_feature_map(s,
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
                               kernel_length=7)

syllable = np.array(['ax'])


example_starts, example_ends = gad.get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               flts,
                                                               log_part_blocks,
                                                               F.shape[0])

avg_bgd = gad.AverageBackground()
s = np.load(train_data_path + file_indices[1]+'s.npy')
F2 = gad.get_waliji_feature_map(s,
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
                               kernel_length=7)

avg_bgd.add_frames(F,time_axis=0)
assert np.all(avg_bgd.E == F.mean(0))
avg_bgd.add_frames(F2,time_axis=0)

assert np.all(avg_bgd.E == np.vstack((F,F2)).mean(0))


file_indices = [f for f in file_indices if f != '']

avg_bgd, syllable_examples = gad.get_syllable_examples_backgrounds_files(train_data_path,file_indices,np.array(['aa','r']),
                                                                         log_part_blocks,
                                                                         log_invpart_blocks,
                                                                         verbose=10)

clipped_bgd = np.clip(avg_bgd.E,.1,.4).astype(np.float32)

np.save(tmp_data_path+'clipped_train_bgd.npy',clipped_bgd)

def get_example_lengths(syllable_examples):
    return np.array(
        tuple(
            np.uint16(x.shape[0]) for x in syllable_examples))

aar_lengths = get_example_lengths(syllable_examples)
np.save(tmp_data_path+'aar_lengths.npy',aar_lengths)

def extend_example_to_max(syllable_example,clipped_bgd,max_length):
    if syllable_example.shape[0] >= max_length:
        return syllable_example.astype(np.uint8)
    else:
        return np.vstack((syllable_example,
                          np.tile(clipped_bgd,
                                  (max_length-syllable_example.shape[0],1,1)))).astype(np.uint8)




def extend_examples_to_max(clipped_bgd,syllable_examples,aar_lengths):
    max_length = aar_lengths.max()
    return np.array(tuple(
        extend_example_to_max(syllable_example,clipped_bgd,max_length)
        for syllable_example in syllable_examples))

aar_examples = extend_examples_to_max(clipped_bgd,syllable_examples,aar_lengths)

np.save(tmp_data_path+'aar_examples.npy',aar_examples)

# time to check that the examples are looking reasonable, particularly
# in the context of the parts
# we want to find the first example that was recorded

def get_first_example(file_indices,train_data_path,syllable):
    for i, idx in enumerate(file_indices):
        phns = np.load(train_data_path+idx+'phns.npy')
        matches = gad.phns_syllable_matches(phns,syllable)
        if len(matches) >0:
            return idx
    return None

syllable = np.array(['aa','r'])
data_file_idx = get_first_example(file_indices,train_data_path,syllable)


s = np.load(data_path+'Train/'+data_file_idx+'s.npy')
phns = np.load(data_path+'Train/'+data_file_idx+'phns.npy')
# we divide by 5 since we coarsen in the time domain
flts = np.load(data_path+'Train/'+data_file_idx+'feature_label_transitions.npy')

abst_threshold=np.array([.025,.025,.015,.015,
                          .02,.02,.02,.02])
spread_length=3
fft_length=512
num_window_step_samples=80
freq_cutoff=3000
sample_rate=16000
num_window_samples=320
kernel_length=7


S = gad.esp.get_spectrogram_features(s,
                                     sample_rate,
                                     num_window_samples,
                                     num_window_step_samples,
                                     fft_length,
                                     freq_cutoff,
                                     kernel_length,
                                 )
E, edge_feature_row_breaks,\
        edge_orientations = gad.esp._edge_map_no_threshold(S)
gad.esp._edge_map_threshold_segments(E,
                                 40,
                                 1,
                                 threshold=.7,
                                 edge_orientations = edge_orientations,
                                 edge_feature_row_breaks = edge_feature_row_breaks)

import matplotlib.pyplot as plt
plt.imshow(S[::-1],interpolation='nearest')
plt.show()

match_locs = gad.phns_syllable_matches(phns,syllable)
match_start_frame = flts[7]
match_end_frame = flts[7+len(syllable)]

plt.close()
plt.imshow(S[::-1,match_start_frame:match_end_frame],interpolation='nearest')
plt.show()


plt.close()
ax = plt.figure()
plt.imshow(S[::-1],interpolation='nearest')
axes = plt.gca()
axes.set_xticks([match_start_frame,match_end_frame])
axes.set_xticklabels(['s','e'])
plt.show()


plt.close()
ax = plt.figure()
plt.imshow(E[::-1],interpolation='nearest')
axes = plt.gca()
axes.set_xticks([match_start_frame,match_end_frame])
axes.set_xticklabels(['s','e'])
plt.show()

plt.close()
ax = plt.figure()
plt.imshow(E[::-1,match_start_frame:match_end_frame],
           interpolation='nearest',aspect=.3)
axes = plt.gca()
plt.show()



E = gad.reorg_part_for_fast_filtering(E)
F = gad.cp.code_parts_fast(E.astype(np.uint8),log_part_blocks,log_invpart_blocks,10)
F = np.argmax(F,2)
    # the amount of spreading to do is governed by the size of the part features

filtsize=3

a = np.arange(100).reshape(10,10)
b = np.lib.stride_tricks.as_strided(a,shape=(a.size,3),strides=(a.itemsize,a.itemsize))
c1 = np.hstack(tuple(
        b[10*i:b.shape[0]-(2-i)*10] for i in xrange(3)))
c_idx = (np.arange((10-3+1)*(10-3+1)) % (10-3+1)) + np.arange((10-3+1)*(10-3+1))/(10-3+1)*10

c = c1[c_idx]
num_windows,window_size = c.shape
c = c.ravel()
d = np.zeros((num_windows,100))
d[np.arange(c.shape[0])/window_size,c] = 1

F = gad.swp.spread_waliji_patches(F,
                                  log_part_blocks.shape[1],
                                  log_part_blocks.shape[2],
                                  log_part_blocks.shape[0])
example_starts, example_ends = gad.get_examples_from_phns_ftls(syllable,
                                                               phns,
                                                               flts,
                                                               log_part_blocks,
                                                               F.shape[0])
G = collapse_to_grid(F,log_part_blocks.shape[1],
                            log_part_blocks.shape[2])
F[example_starts[0]:example_ends[0]][0]
