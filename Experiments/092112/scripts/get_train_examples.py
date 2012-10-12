import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
#root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092112/'
tmp_data_path = exp_path+'data/'
scripts_path = exp_path+'scripts/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
import template_speech_rec.template_experiments as t_exp

from collections import defaultdict


s_fnames = [data_path+'Train/'+str(i+1)+'s.npy' for i in xrange(4619)]
flts_fnames = [data_path+'Train/'+str(i+1)+'feature_label_transitions.npy' for i in xrange(4619)]
phns_fnames = [data_path+'Train/'+str(i+1)+'phns.npy' for i in xrange(4619)]

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


def phns_syllable_matches(phns,syllable):
    syllable_len = len(syllable)
    out = []
    for phn_id in xrange(len(phns)-syllable_len+1):
        if np.all(phns[phn_id:phn_id+syllable_len]==syllable):
            out.append(phn_id)
    return out


avg_bgd_obj = t_exp.AverageBackground()

for s_idx, s_fname in enumerate(s_fnames):
    print "s_idx = %d" %s_idx
    s = np.load(s_fname)
    phns = np.load(phns_fnames[s_idx])
    flts = np.load(flts_fnames[s_idx])
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
    avg_bgd_obj.add_frames(E)
    syllable_locs = [phns_syllable_matches(phns,syllable) for syllable in syllables]
    for syllable_id, syllable_loc_list in enumerate(syllable_locs):
        syllable = tuple(syllables[syllable_id])
        example_E_dict[syllable].extend([
            E[:,max(flts[loc]-offset,0):min(flts[loc+len(syllable)]+offset,E.shape[1])].astype(np.uint8)
            for loc in syllable_loc_list])
        example_S_dict[syllable].extend([
            S[:,max(flts[loc]-offset,0):min(flts[loc+len(syllable)]+offset,E.shape[1])].astype(np.float32)
            for loc in syllable_loc_list])


out = open(tmp_data_path+'example_E_dict.pkl','wb')
cPickle.dump(example_E_dict,out)
out.close()

out = open(tmp_data_path+'example_S_dict.pkl','wb')
cPickle.dump(example_S_dict,out)
out.close()

np.save(tmp_data_path+'avg_E_bgd.npy',avg_bgd_obj.E)






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
        if bps.shape[0] > 300000:
            break
    np.save(tmp_data_path+'bps_%d.npy' % lower_cutoff,bps)
    np.save(tmp_data_path+'spec_windows_%d.npy' % lower_cutoff,spec_windows)
    np.save(tmp_data_path+'s_idx_list_%d.npy' % lower_cutoff,np.array(s_idx_list))


# Now we perform estimation of the parts
for lower_cutoff in [10,30,40,50,60]:
    print "lower_cutoff %d" % lower_cutoff
    bps = np.load(tmp_data_path+'bps_%d.npy' %lower_cutoff)
    spec_windows = np.load(tmp_data_path+'spec_windows_%d.npy' % lower_cutoff)
    s_idx_list = np.load(tmp_data_path+'s_idx_list_%d.npy' % lower_cutoff)
    for num_parts in [15,20,30,50,80,100]:
        print "num_parts = %d" % num_parts
        bm = bernoulli_mixture.BernoulliMixture(num_parts,bps)
        bm.run_EM(.000001,min_probability=.01)
        np.save(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts),bm.templates)
        np.save(tmp_data_path+'bm_affinities%d_%d.npy' % (lower_cutoff,num_parts),bm.affinities)
        bm_templates = bm.templates
        bm_affinities = bm.affinities
        del bm
        spec_avg_parts = np.zeros((num_parts,patch_height+1,patch_width+1),dtype=np.float32)
        spec_weight_sums = np.zeros(num_parts)
        s_idx = -1
        for i,affinity in enumerate(bm_affinities):
            if s_idx_list[i] > s_idx:
                s_idx  = s_idx_list[i]
                S_patches = np.load(tmp_data_path+str(s_idx)+'spec_patch_%d.npy' %lower_cutoff)
                S_patch_start_idx = s_idx
            for part_id, part_affinity in enumerate(affinity):
                if part_affinity > .95:
                    spec_weight_sums[part_id] += part_affinity
                    spec_avg_parts[part_id] += (part_affinity* (S_patches[s_idx-S_patch_start_idx] -  spec_avg_parts[part_id]))/spec_weight_sums[part_id]
        np.save(tmp_data_path+'spec_avg_parts%d_%d.npy' % (lower_cutoff,num_parts), spec_avg_parts)
        np.save(tmp_data_path+'spec_weight_sums%d_%d.npy' % (lower_cutoff,num_parts), spec_weight_sums)


# need to see whether what I'm doing is reasonable at all.
# load in an S
s = np.load(s_fnames[0])
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
upper_cutoff = 10000
lower_cutoff = 30
bp,all_patch_rows,all_patch_cols = elf.extract_local_features_tied(E,patch_height,
                                                                           patch_width, lower_cutoff,
                                                                           upper_cutoff,
                                                                           edge_feature_row_breaks,
            )
# get rid of those that are just hugging the border
use_indices = np.logical_and(all_patch_rows < (E.shape[0] - patch_height),
all_patch_cols < (E.shape[1] - patch_width))
bp = bp[use_indices]
all_patch_rows = all_patch_rows[use_indices]
all_patch_cols = all_patch_cols[use_indices]

patch_row_indices = elf.generate_patch_row_indices(all_patch_rows,patch_height+1,patch_width+1)-1
patch_col_indices = elf.generate_patch_col_indices(all_patch_cols,patch_height+1,patch_width+1)
# np.save(tmp_data_path+str(s_idx)+'all_patch_cols_%d.npy' %lower_cutoff,all_patch_cols)
# s_windows = np.lib.stride_tricks.as_strided(s,shape=(len(s)-patch_width*num_window_samples,patch_width*num_window_samples),
#                                        strides=(s.itemsize,s.itemsize))
# np.save(tmp_data_path+str(s_idx)+'bp_%d.npy' %lower_cutoff,bp)
spec_patches = S[patch_row_indices,patch_col_indices].reshape(bp.shape[0],patch_height+1,patch_width+1)
np.save(tmp_data_path+'spec_patches_ex.npy',spec_patches)
# keeps track of which parts are the spectrogram are being
# used for the patches
S_ones = np.zeros(S.shape,dtype=np.uint8)
S_ones[patch_row_indices,patch_col_indices] = 1
np.save(tmp_data_path+str(s_idx)+'spec_patch_ones_%d.npy' %lower_cutoff,S_ones)


from collections import defaultdict

# Now we perform estimation of the parts
for lower_cutoff in [10,30,40,50,60]:
    print "lower_cutoff %d" % lower_cutoff
    s_idx_list = np.load(tmp_data_path+'s_idx_list_%d.npy' % lower_cutoff)
    for num_parts in [15,20,30,50,80,100]:
        print "num_parts = %d" % num_parts
        bm_templates = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))
        bm_affinities = np.load(tmp_data_path+'bm_affinities%d_%d.npy' % (lower_cutoff,num_parts))
        spec_avg_parts = np.zeros((num_parts,patch_height+1,patch_width+1))
        spec_weight_sums = np.zeros(num_parts)
        part_dict = defaultdict(list)
        s_idx = -1
        for i,affinity in enumerate(bm_affinities):
            if s_idx_list[i] > s_idx:
                s_idx  = s_idx_list[i]
                S_patches = np.load(tmp_data_path+str(s_idx)+'spec_patch_%d.npy' %lower_cutoff)
                S_patch_start_idx = s_idx
            for part_id, part_affinity in enumerate(affinity):
                if part_affinity > .95:
                    spec_weight_sums[part_id] += part_affinity
                    spec_avg_parts[part_id] += (part_affinity* (S_patches[s_idx-S_patch_start_idx] -  spec_avg_parts[part_id]))/spec_weight_sums[part_id]
                    part_dict[part_id].append(S_patches[s_idx-S_patch_start_idx])
        np.save(tmp_data_path+'spec_avg_parts%d_%d.npy' % (lower_cutoff,num_parts), spec_avg_parts)
        np.save(tmp_data_path+'spec_weight_sums%d_%d.npy' % (lower_cutoff,num_parts), spec_weight_sums)
        for i in xrange(bm_affinities.shape[1]):
            np.save(tmp_data_path+'part_records%d_%d_%d' % (i,lower_cutoff,num_parts),part_dict[i])


for lower_cutoff in [40]:
    print "lower_cutoff %d" % lower_cutoff
    s_idx_list = np.load(tmp_data_path+'s_idx_list_%d.npy' % lower_cutoff)
    for num_parts in [30]:
        print "num_parts = %d" % num_parts
        bm_templates = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))
        bm_affinities = np.load(tmp_data_path+'bm_affinities%d_%d.npy' % (lower_cutoff,num_parts))
        spec_avg_parts = np.zeros((num_parts,patch_height+1,patch_width+1))
        spec_weight_sums = np.zeros(num_parts)
        part_dict = defaultdict(list)
        s_idx = -1
        for i,affinity in enumerate(bm_affinities):
            if s_idx_list[i] > s_idx:
                s_idx  = s_idx_list[i]
                S_patches = np.load(tmp_data_path+str(s_idx)+'spec_patch_%d.npy' %lower_cutoff)
                S_patch_start_idx = s_idx
            for part_id, part_affinity in enumerate(affinity):
                if part_affinity > .95:
                    spec_weight_sums[part_id] += part_affinity
                    spec_avg_parts[part_id] += (part_affinity* (S_patches[s_idx-S_patch_start_idx] -  spec_avg_parts[part_id]))/spec_weight_sums[part_id]
                    part_dict[part_id].append(S_patches[s_idx-S_patch_start_idx])
        np.save(tmp_data_path+'spec_avg_parts%d_%d.npy' % (lower_cutoff,num_parts), spec_avg_parts)
        np.save(tmp_data_path+'spec_weight_sums%d_%d.npy' % (lower_cutoff,num_parts), spec_weight_sums)
        for i in xrange(bm_affinities.shape[1]):
            np.save(tmp_data_path+'part_records%d_%d_%d' % (i,lower_cutoff,num_parts),part_dict[i])







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



#
#
# Go through the files again checking whether the parts picked up by the mixture models make any sense
#

#
#
# running this again, this time trying to capture exactly what is happening in the spec_windows
#

for lower_cutoff in [10,30,40,50,60]:
    print "lower_cutoff = %d" %lower_cutoff
    bps = np.zeros((0,40,5),dtype=np.uint8)
    spec_windows = np.zeros((0,6,6),dtype=np.float32)
    s_idx_list = []
    for s_idx, s_fname in enumerate(s_fnames[:400]):
        print "s_idx = %d" %s_idx
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
        s_idx_list.extend(bp.shape[0] *[s_idx])
        # need to capture the fact that we have bigger patches for the spectrogram
        patch_row_indices = elf.generate_patch_row_indices(all_patch_rows,patch_height+1,patch_width+1)-1
        patch_col_indices = elf.generate_patch_col_indices(all_patch_cols,patch_height+1,patch_width+1)
        bps = np.vstack((bps,bp.astype(np.uint8)))
        spec_windows = np.vstack((spec_windows,S[patch_row_indices,patch_col_indices].reshape(bp.shape[0],patch_height+1,patch_width+1).astype(np.float32)))
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
        if bps.shape[0] > 100000:
            break
    np.save(tmp_data_path+'bps10k_%d.npy' % lower_cutoff,bps)
    np.save(tmp_data_path+'spec_windows10k_%d.npy' % lower_cutoff,spec_windows)
    np.save(tmp_data_path+'s_idx_list_%d.npy' % lower_cutoff,np.array(s_idx_list))


# Now we perform estimation of the parts
# everything is labeled 10k but its really 100k
for lower_cutoff in [10,30,40,50,60]:
    print "lower_cutoff %d" % lower_cutoff
    bps = np.load(tmp_data_path+'bps10k_%d.npy' %lower_cutoff)
    spec_windows = np.load(tmp_data_path+'spec_windows10k_%d.npy' % lower_cutoff)
    for num_parts in [15,20,30,50,80,100]:
        print "num_parts = %d" % num_parts
        bm = bernoulli_mixture.BernoulliMixture(num_parts,bps)
        bm.run_EM(.000001,min_probability=.01)
        np.save(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts),bm.templates)
        np.save(tmp_data_path+'bm_affinities%d_%d.npy' % (lower_cutoff,num_parts),bm.affinities)
        spec_avgs = np.dot(bm.affinities.T,spec_windows.reshape(bm.affinities.shape[0],36))
        np.save(tmp_data_path+'spec_avgs10k_%d_%d.npy' %(lower_cutoff,num_parts),spec_avgs.reshape(bm.affinities.shape[1],6,6))
