import numpy as np
import pylab as pl
from sklearn import svm
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import template_speech_rec.code_parts as cp
import template_speech_rec.spread_waliji_patches as swp
import matplotlib.pyplot as plt

import pickle,collections

sp = gtrd.SpectrogramParameters(
    sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False)

ep = gtrd.EdgemapParameters(block_length=40,
                            spread_length=1,
                            threshold=.7)


#
# need to access the files where we perform the estimation
#

utterances_path = '/home/mark/Template-Speech-Recognition/Data/Train/'
file_indices = gtrd.get_data_files_indices(utterances_path)

inner_mask = np.zeros((5,5),dtype=np.uint8)
inner_mask[1:-1,1:-1] = 1
all_patches = np.zeros((0,5,5,8),dtype=np.uint8)
all_S_patches = np.zeros((0,5,5),dtype=np.float32)
inner_thresh = 9
outer_thresh = 40

for fl_idx,fl in enumerate(file_indices[:50]):
    utterance = gtrd.makeUtterance(utterances_path,fl)
    S = gtrd.get_spectrogram(utterance.s,sp)
    E = gtrd.get_edge_features(S.T,ep,verbose=False)
    patch_set, S_patch_set, patch_counts, patch_locs = cp.get_parts_mask(E.astype(np.uint8),
                                                                         inner_mask,
                                                                         S.astype(np.float32),
                                                                         inner_thresh)
    use_patch_ids = patch_set.sum(-1).sum(-1).sum(-1) >= outer_thresh
    all_patches = np.vstack(
        (all_patches,
         patch_set[use_patch_ids ]))
    all_S_patches = np.vstack(
        (all_S_patches,
         S_patch_set[use_patch_ids].astype(np.float32)))

    
import template_speech_rec.bernoulli_mixture as bm

np.save('data/all_patches',all_patches)
np.save('data/all_S_patches',all_S_patches)

num_parts_list = [20,40,60,80,100,150,200]
for num_parts in num_parts_list:
    bem = bm.BernoulliMixture(num_parts,all_patches)
    bem.run_EM(.00001)
    np.save('parts_templates_%d_%d_%d_%d' % (num_parts,inner_thresh,outer_thresh,all_patches.shape[0]),
                                   bem.templates)
    np.save('parts_affinities_%d_%d_%d_%d' % (num_parts,inner_thresh,outer_thresh,all_patches.shape[0]),
                                   bem.affinities)
    parts_shape = all_S_patches.shape
    spec_part_templates = np.dot(bem.affinities.T,all_S_patches.reshape(
            parts_shape[0],
            np.prod(parts_shape[1:]))).reshape(
        (bem.affinities.shape[1],) + parts_shape[1:])
    np.save('spec_part_templates_%d_%d_%d_%d' % (num_parts,inner_thresh,outer_thresh,all_patches.shape[0]),spec_part_templates)


fl_idx = 0
fl = file_indices[fl_idx]
utterance = gtrd.makeUtterance(utterances_path,fl)



S = gtrd.get_spectrogram(utterance.s,sp)
E = gtrd.get_edge_features(S.T,ep,verbose=False)


patch_set, S_patch_set, patch_counts, patch_locs = cp.get_parts_mask(E.astype(np.uint8),
                                           np.ones((5,5),dtype=np.uint8),
                                           S.astype(np.float32),
                                           18)





for fl_id,fl in enumerate(file_indices[:10]


outfile = np.load('data/aar_Es_lengths.npz')
Es= outfile['arr_0']
lengths=outfile['arr_1']
example_mat=outfile['arr_2']
del outfile


