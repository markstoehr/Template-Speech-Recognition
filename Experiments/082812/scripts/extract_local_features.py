import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
#root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/082812/'
tmp_data_path = exp_path+'data/'
scripts_path = exp_path+'scripts/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
#
# get the signal procressing loop
#

s_fnames = [data_path+'Train/'+str(i+1)+'s.npy' for i in xrange(4619)]

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

s_idx =0
s_fname = s_fnames[s_idx]

bps = np.zeros((0,40,5),dtype=np.uint8)
patch_s_windows = np.zeros((0,1600))
spec_windows = np.zeros((0,6,6))
f = open(tmp_data_path+'guide_to_tmp_data.txt','w')
s_idx_list = []
for s_idx, s_fname in enumerate(s_fnames[:400]):
    f.write(str(s_idx)+'\t'+s_fname+'\n')
    s = np.load(s_fname)
    S = esp.get_spectrogram_features(s,
                                     sample_rate,
                                     num_window_samples,
                                     num_window_step_samples,
                                     fft_length,
                                     freq_cutoff,
                                     kernel_length)
    E, edge_feature_row_breaks,\
      edge_orientations = esp._edge_map_no_threshold(S)
    esp._edge_map_threshold_segments(E,
                                     40,
                                     1,
                                     threshold=.3,
                                     edge_orientations = edge_orientations,
                                     edge_feature_row_breaks = edge_feature_row_breaks)
    patch_width = 5
    patch_height = 5
    lower_quantile = .9
    upper_quantile = 1.
    bp,all_patch_rows,all_patch_cols = elf.extract_local_features_tied(E,patch_height,
                                                                       patch_width, lower_quantile,
                                                                       upper_quantile,
                                                                       edge_feature_row_breaks,
                                                                       segment_ms=500,
                                                                       hop_ms = 5)
    bps = np.vstack((bps,bp.astype(np.uint8)))
    s_idx_list.extend(bp.shape[0] *[s_idx])
    # need to capture the fact that we have bigger patches for the spectrogram
    patch_row_indices = elf.generate_patch_row_indices(all_patch_rows,patch_height+1,patch_width+1)-1
    patch_col_indices = elf.generate_patch_col_indices(all_patch_cols,patch_height+1,patch_width+1)
    np.save(tmp_data_path+str(s_idx)+'all_patch_cols.npy',all_patch_cols)
    s_windows = np.lib.stride_tricks.as_strided(s,shape=(len(s)-patch_width*num_window_samples,patch_width*num_window_samples),
                                                strides=(s.itemsize,s.itemsize))
    patch_s_windows = s_windows[all_patch_cols * 80]
    np.save(tmp_data_path+str(s_idx)+'bp.npy',bp)
    np.save(tmp_data_path+str(s_idx)+'patch_s_windows.npy',patch_s_windows)
    np.save(tmp_data_path+str(s_idx)+'spec_patch.npy',S[patch_row_indices,patch_col_indices].reshape(bp.shape[0],patch_height+1,patch_width+1))
    # keeps track of which parts are the spectrogram are being
    # used for the patches
    S_ones = np.zeros(S.shape,dtype=np.uint8)
    S_ones[patch_row_indices,patch_col_indices] = 1
    np.save(tmp_data_path+str(s_idx)+'spec_patch_ones.npy',S_ones)


f.close()
np.save(tmp_data_path+'s_idx_list.npy',np.array(s_idx_list))


bp=np.load(tmp_data_path+'1bp.npy',bp).astype(np.uint8)
bm = bernoulli_mixture.BernoulliMixture(20,bp)
bm.run_EM(.000001,min_probability=.001)
np.save(tmp_data_path+'1bm_templates%d.npy' % num_parts,bm.templates)
np.save(tmp_data_path+'1bm_affinities%d.npy' % num_parts,bm.affinities)
bm_templates = bm.templates
bm_affinities = bm.affinities
del bm
spec_avg_parts = np.zeros((num_parts,patch_height+1,patch_width+1))
spec_weight_sums = np.zeros(num_parts)
s_idx = -1
for i,affinity in enumerate(bm_affinities):
    if s_idx_list[i] > s_idx:
        s_idx  = s_idx_list[i]
        S_patches = np.load(tmp_data_path+str(s_idx)+'spec_patch.npy')
        S_patch_start_idx = s_idx
    for part_id, part_affinity in enumerate(affinity):
        if part_affinity > .95:
            spec_weight_sums[part_id] += part_affinity
            spec_avg_parts[part_id] += (part_affinity* (S_patches[s_idx-S_patch_start_idx] -  spec_avg_parts[part_id]))/spec_weight_sums[part_id]
    np.save(tmp_data_path+'1spec_avg_parts%d.npy' % num_parts, spec_avg_parts)



np.save(tmp_data_path+'bps.npy',bps)
bps = np.load(tmp_data_path+'bps.npy')



for num_parts in [10,20,30,50,80,100]:
    print "Working on number of parts is %d" % num_parts
    bm = bernoulli_mixture.BernoulliMixture(num_parts,bps[:50000])
    bm.run_EM(.000001,min_probability=.001)
    np.save(tmp_data_path+'bm_templates%d.npy' % num_parts,bm.templates)
    np.save(tmp_data_path+'bm_affinities%d.npy' % num_parts,bm.affinities)
    bm_templates = bm.templates
    bm_affinities = bm.affinities
    del bm
    spec_avg_parts = np.zeros((num_parts,patch_height+1,patch_width+1))
    spec_weight_sums = np.zeros(num_parts)
    s_idx = -1
    for i,affinity in enumerate(bm_affinities):
        if s_idx_list[i] > s_idx:
            s_idx  = s_idx_list[i]
            S_patches = np.load(tmp_data_path+str(s_idx)+'spec_patch.npy')
            S_patch_start_idx = s_idx
        for part_id, part_affinity in enumerate(affinity):
            if part_affinity > .95:
                spec_weight_sums[part_id] += part_affinity
                spec_avg_parts[part_id] += (part_affinity* (S_patches[s_idx-S_patch_start_idx] -  spec_avg_parts[part_id]))/spec_weight_sums[part_id]
    np.save(tmp_data_path+'spec_avg_parts%d.npy' % num_parts, spec_avg_parts)



for num_parts in [15,20,25]:
    print "Working on number of parts is %d" % num_parts
    bm = bernoulli_mixture.BernoulliMixture(num_parts,bps[:100000])
    bm.run_EM(.000001,min_probability=.001)
    np.save(tmp_data_path+'100kbm_templates%d.npy' % num_parts,bm.templates)
    np.save(tmp_data_path+'100kbm_affinities%d.npy' % num_parts,bm.affinities)
    bm_templates = bm.templates
    bm_affinities = bm.affinities
    del bm
    spec_avg_parts = np.zeros((num_parts,patch_height+1,patch_width+1))
    spec_weight_sums = np.zeros(num_parts)
    s_idx = -1
    for i,affinity in enumerate(bm_affinities):
        if s_idx_list[i] > s_idx:
            s_idx  = s_idx_list[i]
            S_patches = np.load(tmp_data_path+str(s_idx)+'spec_patch.npy')
            S_patch_start_idx = s_idx
        for part_id, part_affinity in enumerate(affinity):
            if part_affinity > .95:
                spec_weight_sums[part_id] += part_affinity
                spec_avg_parts[part_id] += (part_affinity* (S_patches[s_idx-S_patch_start_idx] -  spec_avg_parts[part_id]))/spec_weight_sums[part_id]
    np.save(tmp_data_path+'100kspec_avg_parts%d.npy' % num_parts, spec_avg_parts)
    np.save(tmp_data_path+'100kspec_weight_sums%d.npy' % num_parts, spec_weight_sums)


print 'hi'
# now writing a function for coding parts



