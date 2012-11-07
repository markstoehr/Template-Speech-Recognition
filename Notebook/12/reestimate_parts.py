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

fl = file_indices[49]
if True:
    utterance = gtrd.makeUtterance(utterances_path,fl)
    S = gtrd.get_spectrogram(utterance.s,sp)
    E = gtrd.get_edge_features(S.T,ep,verbose=False)


    
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


import matplotlib.cm as cm
for num_parts in [20,40,60]:
    plt.figure()
    plt.clf()
    spec_part_templates = np.load('spec_part_templates_%d_%d_%d_%d.npy' % (num_parts,inner_thresh,outer_thresh,193474))
    for i in xrange(num_parts):
        plt.subplot(9,9,i+1)
        plt.imshow(spec_part_templates[i],cmap=cm.bone)
        plt.axis('off')
    plt.savefig('spec_part_templates_%d.png' % num_parts)

# now do the coding of the true and the false positives
# first check that the parts make sense graphically

import matplotlib.cm as cm
inner_thresh=9
outer_thresh=40
num_parts = 20

M = np.zeros((5,5),dtype=np.uint8)
M[1:-1,1:-1] = 1
templates = np.load('parts_templates_%d_%d_%d_%d.npy' % (num_parts,inner_thresh,outer_thresh,193474))
log_parts = np.log(templates).astype(np.float32)
log_inv_parts = np.log(1-templates).astype(np.float32)
spec_part_templates = np.load('spec_part_templates_%d_%d_%d_%d.npy' % (num_parts,inner_thresh,outer_thresh,193474))

out_map = cp.code_parts_mask(E.astype(np.uint8),M,log_parts,log_inv_parts,inner_thresh,outer_thresh)
out_map_max = np.argmax(out_map,2)
for i in xrange(spec_part_templates.shape[0]):
    plt.figure()
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(S.T,cmap=cm.bone,aspect=3)
    nz_idx = np.arange(out_map_max.size)[(out_map_max == i+1).ravel()]
    x_nz_idx = nz_idx / S.shape[1]
    y_nz_idx = nz_idx % S.shape[1]
    plt.scatter(x_nz_idx,y_nz_idx,c='r')
    plt.axis('off')
    plt.subplot(2,1,2)
    plt.imshow(spec_part_templates[i],cmap=cm.bone)
    plt.axis('off')
    plt.savefig('template_check_%d_%d.png' % (num_parts,i))
    plt.close()


for i in xrange(spec_part_templates.shape[0]):
    print i
    num_matches = np.sum(out_map_max == i+1)
    num_cols = int(np.sqrt(2*num_matches+4))
    num_rows = int(np.ceil(2*(num_matches+2)/float(num_cols)))
    plt.figure()
    plt.clf()
    plt.title('Number of parts = %d, part index = %d'%(num_parts,i))
    plt.subplot(num_rows,num_cols,1)
    plt.imshow(spec_part_templates[i],cmap=cm.bone,interpolation='nearest')
    plt.axis('off')
    raw_match_idx = np.arange(out_map_max.size)[(out_map_max==i+1).ravel()]
    j=0
    for j in xrange(min(raw_match_idx.shape[0],40)):
        if j % 10 == 0: print j
        plt.subplot(num_rows,num_cols,2+j)
        start_i = raw_match_idx[j]/out_map_max.shape[1]
        start_j = raw_match_idx[j]%out_map_max.shape[1]
        plt.imshow(S[start_i:start_i+5,
                     start_j:start_j+5],cmap=cm.bone,interpolation='nearest')
        plt.axis('off')
    j+=1
    plt.subplot(num_rows,num_cols,j+2)
    plt.imshow(templates[i].swapaxes(2,0).reshape(40,5),interpolation='nearest')
    plt.axis('off')
    j+=1
    for k in xrange(j,j+min(raw_match_idx.shape[0],40)):
        if k % 10 == 0: print j
        plt.subplot(num_rows,num_cols,2+k)
        start_i = raw_match_idx[k-j]/out_map_max.shape[1]
        start_j = raw_match_idx[k-j]%out_map_max.shape[1]
        plt.imshow(E[start_i:start_i+5,
                     start_j:start_j+5].swapaxes(2,0).reshape(40,5),interpolation='nearest')
        plt.axis('off')
    plt.savefig('template_matches_%d_%d.png' % (num_parts,i))
    plt.close()


# looking at part 15,

patch_idx_detected_by3 = (np.arange(out_map_max.size)[(out_map_max==4).ravel()])

is_x = patch_idx_detected_by3/out_map_max.shape[1]
is_y = patch_idx_detected_by3 % out_map_max.shape[1]

S_patches_by_3 = np.array([
        S[is_x[i]:is_x[i]+5,
          is_y[i]: is_y[i]+5]
        for i in xrange(len(patch_idx_detected_by3))])

n = len(S_patches_by_3)
num_rows = int(np.ceil(np.sqrt(n)))
num_cols = num_rows
plt.close()
plt.figure()
plt.clf()
for i in xrange(n):
    plt.subplot(num_rows,num_cols,i+1)
    plt.imshow(S_patches_by_3[i])
    plt.axis('off')

plt.show()
plt.close()

patches_by_8 = np.zeros(
    (len(patch_idx_detected_by8),
     5,5,8),dtype=np.uint8)
for j,i in enumerate(patch_idx_detected_by8):
    patches_by_8[j][:] = E[i/out_map.shape[1]: i/out_map.shape[1]+5,
          i%out_map.shape[1]: (i%out_map.shape[1])+5,:]


patches_by_8 = np.array([
        E[i/S.shape[1]: i/S.shape[1]+5,
          i%S.shape[1]: (i%S.shape[1])+5,:]


S_patches_by_8 = np.array([
        S[i/S.shape[1]: i/S.shape[1]+5,
          i%S.shape[1]: i%S.shape[1]+5]
        for i in patch_idx_detected_by8])


out_map = np.argmax(out_map,2)
    # the amount of spreading to do is governed by the size of the part features
out_map = swp.spread_waliji_patches(out_map,
                                  log_parts.shape[1],
                                  log_parts.shape[2],
                                  log_parts.shape[0])
out_map = gtrd.collapse_to_grid(out_map,log_parts.shape[1],
                         log_parts.shape[2])




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


#
# Trying different sized templates
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

fl = file_indices[49]
if True:
    utterance = gtrd.makeUtterance(utterances_path,fl)
    S = gtrd.get_spectrogram(utterance.s,sp)
    E = gtrd.get_edge_features(S.T,ep,verbose=False)


    
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


import matplotlib.cm as cm
for num_parts in [20,40,60]:
    plt.figure()
    plt.clf()
    spec_part_templates = np.load('spec_part_templates_%d_%d_%d_%d.npy' % (num_parts,inner_thresh,outer_thresh,193474))
    for i in xrange(num_parts):
        plt.subplot(9,9,i+1)
        plt.imshow(spec_part_templates[i],cmap=cm.bone)
        plt.axis('off')
    plt.savefig('spec_part_templates_%d.png' % num_parts)



