import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import matplotlib.pyplot as plt
import cPickle,os, pickle,collections

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


utterances_path = '/home/mark/Template-Speech-Recognition/Data/Train/'
file_indices = gtrd.get_data_files_indices(utterances_path)

phns = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 'sp', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'sil']

phn = ('aa')

num_mix_params = [1,2,3,5,7,9]

# timit mapping
leehon_mapping_base = { 'aa':'aa',
                   'ao':'aa',
                   'ax':'ah',
                   'ax-h':'ah',
                   'axr': 'er',
                   'hv':'hh',
                   'ix':'ih',
                   'el':'l',
                   'em':'m',
                   'en':'n',
                   'nx':'n',
                   'eng':'ng',
                   'zh':'sh',
                   'ux':'uw',
                   'pcl':'sil',
                   'tcl':'sil',
                   'kcl':'sil',
                   'bcl':'sil',
                   'dcl':'sil',
                   'gcl':'sil',
                   'h#':'sil',
                   'pau':'sil',
                   'epi':'sil',
                   'sp':'sil'}


phns_down = list(set([p for p in phns if p not in leehon_mapping_base.keys() and p != 'q'] + [leehon_mapping_base[p] for p in phns if p in leehon_mapping_base.keys() and p != 'q']))

leehon_mapping = {}
for k,v in leehon_mapping_base.items():
    leehon_mapping[k] = v

for phn in phns_down:
    if phn not in leehon_mapping.keys():
        leehon_mapping[phn] = phn

leehon_mapping["q"] = "q"


use_phns = np.array(list(set(leehon_mapping.values())))
phn_idx = 0
phn = (use_phns[phn_idx],)
test_path = '/var/tmp/stoehr/Template-Speech-Recognition/Data/Train/'
test_file_indices = gtrd.get_data_files_indices(train_path)
test_example_lengths = gtrd.get_detect_lengths(train_file_indices,train_path)
np.save("data/test_example_lengths.npy",test_example_lengths)
np.save("data/test_file_indices.npy",test_file_indices)

for phn_idx in xrange(use_phns.shape[0]):
    phn = (use_phns[phn_idx],)
    phn_features,avg_bgd=gtrd.get_syllable_features_directory(utterances_path,file_indices,phn,
                                                              S_config=sp,E_config=ep,offset=0,
                                                              E_verbose=False,return_avg_bgd=True,
                                                              waveform_offset=15,
                                                              phn_mapping=leehon_mapping)
    bgd = np.clip(avg_bgd.E,.01,.99)
    np.save('data/%s_bgd.npy' %phn,bgd)
    example_mat = gtrd.recover_example_map(phn_features)
    lengths,waveforms  = gtrd.recover_waveforms(phn_features,example_mat)
    np.savez('data/%s_waveforms_lengths.npz' %phn,waveforms=waveforms,
             lengths=lengths,
         example_mat=example_mat)
    Slengths,Ss  = gtrd.recover_specs(phn_features,example_mat)
    Ss = Ss.astype(np.float32)
    np.savez('data/%s_Ss_lengths.npz' %phn,Ss=Ss,Slengths=Slengths,example_mat=example_mat)
    Elengths,Es  = gtrd.recover_edgemaps(phn_features,example_mat,bgd=bgd)
    Es = Es.astype(np.uint8)
    np.savez('data/%s_Es_lengths.npz' %phn,Es=Es,Elengths=Elengths,example_mat=example_mat)
    # the Es are padded from recover_edgemaps
    for num_mix in num_mix_params:
        if num_mix == 1:
            affinities = np.ones(Es.shape[0])
            templates = (np.mean(Es,0),)
            spec_templates = (np.mean(Ss,0),)
            np.save('data/%s_%d_affinities.npy' % (phn,num_mix),
                    bem.affinities)
            np.save('data/%s_%d_templates.npy' % (phn,num_mix),
                    templates)
            np.save('data/%s_%d_spec_templates.npy' % (phn,num_mix),
                    spec_templates)
        else:
            bem = bm.BernoulliMixture(num_mix,Es)
            bem.run_EM(.000001)
            templates = et.recover_different_length_templates(bem.affinities,
                                                              Es,
                                                              lengths)
            spec_templates = et.recover_different_length_templates(bem.affinities,
                                                               Ss,
                                                               Slengths)
            np.save('data/%s_%d_affinities.npy' % (phn,num_mix),
                    bem.affinities)
            np.save('data/%s_%d_templates.npy' % (phn,num_mix),
                    templates)
            np.save('data/%s_%d_spec_templates.npy' % (phn,num_mix),
                    spec_templates)
        
FOMS = collections.defaultdict(list)
for num_mix in num_mix_params:
    templates = (np.load('aar1_templates_%d.npz' % num_mix))['arr_0']
    detection_array = np.zeros((train_example_lengths.shape[0],
                            train_example_lengths.max() + 2),dtype=np.float32)
    linear_filters_cs = et.construct_linear_filters(templates,
                                                    bgd)
    np.savez('data/linear_filter_aar_%d.npy'% num_mix,linear_filters_cs[:][0])
    np.savez('data/c_aar_%d.npy'%num_mix,np.array(linear_filters_cs[:][1]))
    syllable = np.array(['aa','r'])
    detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores_mixture(train_path,
                                                                                                   detection_array,
                                                                                                   syllable,
                                                                                                   linear_filters_cs,
                                                                                                                                                                               verbose=True)
    np.save('data/detection_array_aar_%d.npy' % num_mix,detection_array)
    if num_mix == 2:
        out = open('data/example_start_end_times_aar.pkl','wb')
        cPickle.dump(example_start_end_times,out)
        out.close()
        out = open('data/detection_lengths_aar.pkl','wb')
        cPickle.dump(detection_lengths,out)
        out.close()
    window_start = -10
    window_end = 10
    max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        example_start_end_times,
                                                               detection_lengths,
                                                        window_start,
                                                        window_end)
    np.save('data/max_detect_vals_aar_%d.npy' % num_mix,max_detect_vals)
    C0 = 33
    C1 = int( 33 * 1.5 + .5)
    frame_rate = 1/.005
    fpr, tpr = rf.get_roc_curve(max_detect_vals,
                                detection_array,
                                np.array(detection_lengths),
                        example_start_end_times,
                        C0,C1,frame_rate)
    np.save('data/fpr_aar_%d.npy' % num_mix,
            fpr)
    np.save('data/tpr_aar_%d.npy' % num_mix,
            tpr)
    detection_clusters = rf.get_detect_clusters_threshold_array(max_detect_vals,
                                                                detection_array,
                                                                np.array(detection_lengths),
                                                                C0,C1)
    out = open('data/detection_clusters_aar_%d.npy' % num_mix,
               'wb')
    cPickle.dump(detection_clusters,out)
    out.close()
    for i in xrange(1,11):
        thresh_idx = np.arange(fpr.shape[0])[fpr*60 <= i].min()
        FOMS[num_mix].append(tpr[thresh_idx])



