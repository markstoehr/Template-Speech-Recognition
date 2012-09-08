import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
#root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/090712/'
tmp_data_path = exp_path+'data/'
scripts_path = exp_path+'scripts/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf


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

for lower_cutoff in [10,30,40,50,60]:
    print "lower_cutoff = %d" %lower_cutoff
    bps = np.zeros((0,40,5),dtype=np.uint8)
    spec_windows = np.zeros((0,6,6))
    f = open(tmp_data_path+'guide_to_tmp_data_%d.txt' %lower_cutoff,'w')
    s_idx_list = []
    for s_idx, s_fname in enumerate(s_fnames[:400]):
        print "s_idx = %d" %s_idx
        f.write(str(s_idx)+'\t'+s_fname+'\n')
        s = np.load(s_fname)
        S = esp.get_spectrogram_features(s,
                                         sample_rate,
                                         num_window_samples,
                                         num_window_step_samples,
                                         fft_length,
                                         freq_cutoff,
                                     kernel_length)
        if lower_cutoff == 10:
            np.save(tmp_data_path+str(s_idx)+'S.npy',S)
        E, edge_feature_row_breaks,\
          edge_orientations = esp._edge_map_no_threshold(S)
        esp._edge_map_threshold_segments(E,
                                         20,
                                         1,
                                         threshold=.7,
                                         edge_orientations = edge_orientations,
                                         edge_feature_row_breaks = edge_feature_row_breaks)
        if lower_cutoff == 10:
            np.save(tmp_data_path+str(s_idx)+'E.npy',E)
        patch_width = 5
        patch_height = 5
        upper_cutoff = 200
        bp,all_patch_rows,all_patch_cols = elf.extract_local_features_tied(E,patch_height,
                                                                           patch_width, lower_cutoff,
                                                                           upper_cutoff,
                                                                           edge_feature_row_breaks,
            )
        # get rid of those that are just hugging the border
        use_indices = np.logical_and(all_patch_rows < E.shape[0] - patch_height,
                                     all_patch_cols < E.shape[1] - patch_width)
        bp = bp[use_indices]
        all_patch_rows = all_patch_rows[use_indices]
        all_patch_cols = all_patch_cols[use_indices]
        bps = np.vstack((bps,bp.astype(np.uint8)))
        s_idx_list.extend(bp.shape[0] *[s_idx])
        # need to capture the fact that we have bigger patches for the spectrogram
        patch_row_indices = elf.generate_patch_row_indices(all_patch_rows,patch_height+1,patch_width+1)-1
        patch_col_indices = elf.generate_patch_col_indices(all_patch_cols,patch_height+1,patch_width+1)
        # np.save(tmp_data_path+str(s_idx)+'all_patch_cols_%d.npy' %lower_cutoff,all_patch_cols)
        # s_windows = np.lib.stride_tricks.as_strided(s,shape=(len(s)-patch_width*num_window_samples,patch_width*num_window_samples),
        #                                        strides=(s.itemsize,s.itemsize))
        # np.save(tmp_data_path+str(s_idx)+'bp_%d.npy' %lower_cutoff,bp)
        np.save(tmp_data_path+str(s_idx)+'spec_patch_%d.npy' %lower_cutoff,S[patch_row_indices,patch_col_indices].reshape(bp.shape[0],patch_height+1,patch_width+1).astype(np.float32))
        # keeps track of which parts are the spectrogram are being
        # used for the patches
        S_ones = np.zeros(S.shape,dtype=np.uint8)
        S_ones[patch_row_indices,patch_col_indices] = 1
        np.save(tmp_data_path+str(s_idx)+'spec_patch_ones_%d.npy' %lower_cutoff,S_ones)
    np.save(tmp_data_path+'bps_%d.npy' % lower_cutoff,bps)
    np.save(tmp_data_path+'spec_windows_%d.npy' % lower_cutoff,spec_windows)




bps = np.zeros((0,40,5),dtype=np.uint8)
patch_s_windows = np.zeros((0,1600))
spec_windows = np.zeros((0,6,6))

s_idx_list = []
for s_idx, s_fname in enumerate(s_fnames[:50]):
    print s_idx
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
                                     20,
                                     1,
                                     threshold=.7,
                                     edge_orientations = edge_orientations,
                                     edge_feature_row_breaks = edge_feature_row_breaks)
    patch_width = 5
    patch_height = 5
    lower_cutoff = 0
    upper_cutoff = 200
    bp,all_patch_rows,all_patch_cols = elf.extract_local_features_tied(E,patch_height,
                                                                       patch_width, lower_cutoff,
                                                                       upper_cutoff,
                                                                       edge_feature_row_breaks,
                                                                       )
    bps = np.vstack((bps,bp.astype(np.uint8)))

np.save(tmp_data_path+'bps_sum.npy',bps.sum(1).sum(1))


bps2=np.load(tmp_data_path+'1bp.npy').astype(np.uint8)
S = np.load(tmp_data_path+'1S.npy')
E = np.load(tmp_data_path+'1E.npy')

num_features = E.shape[0]
num_freq_bands = num_features/8
edge_feature_row_breaks = np.arange(9,dtype=int)*num_freq_bands
patch_height, patch_width = parts[0].log_template.shape
patch_height /= 8

bps,all_patch_rows,all_patch_cols = elf.extract_local_features_tied(E,patch_height,
                                                                       patch_width, .9,
                                                                       1.,
                                                                       edge_feature_row_breaks,
                                                                       segment_ms=500,
                                                                       hop_ms = 5)


for num_parts in [15,20,25]:
    print "Working on number of parts is %d" % num_parts
    spec_avg_parts = np.load(tmp_data_path+'100kspec_avg_parts%d.npy' % num_parts).astype(np.float32)
    parts = waliji_code.make_WalijiPart_tuple(np.load(tmp_data_path+'100kbm_templates%d.npy' % num_parts))
    feature_map, S_coded = waliji_code.code_features(E,parts,spec_avg_parts=spec_avg_parts)
    np.save(tmp_data_path+'1S_coded%d.npy' % num_parts,S_coded)
