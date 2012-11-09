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


utterances_path = '/var/tmp/stoehr/Template-Speech-Recognition/Data/Train/'
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
test_path = '/var/tmp/stoehr/Template-Speech-Recognition/Data/Test/'
test_file_indices = gtrd.get_data_files_indices(test_path)
test_example_lengths = gtrd.get_detect_lengths(test_file_indices,test_path)
np.save("data/test_example_lengths.npy",test_example_lengths)
np.save("data/test_file_indices.npy",test_file_indices)


train_path = '/var/tmp/stoehr/Template-Speech-Recognition/Data/Train/'
train_file_indices = gtrd.get_data_files_indices(train_path)
train_example_lengths = gtrd.get_detect_lengths(train_file_indices,train_path)
np.save("data/train_example_lengths.npy",train_example_lengths)
np.save("data/train_file_indices.npy",train_file_indices)

def perform_phn_template_estimation(phn,utterances_path,
                                    file_indices,sp,ep,
                                    num_mix_params,
                                    phn_mapping=None,
                                    waveform_offset=15)
    phn_tuple = (phn,)
    print phn
    phn_features,avg_bgd=gtrd.get_syllable_features_directory(utterances_path,file_indices,phn_tuple,
                                                              S_config=sp,E_config=ep,offset=0,
                                                              E_verbose=False,return_avg_bgd=True,
                                                              waveform_offset=15,
                                                              phn_mapping=leehon_mapping)
    bgd = np.clip(avg_bgd.E,.01,.99)
    np.save('data/bgd.npy',bgd)
    example_mat = gtrd.recover_example_map(phn_features)
    lengths,waveforms  = gtrd.recover_waveforms(phn_features,example_mat)
    np.savez('data/waveforms_lengths.npz',waveforms=waveforms,
             lengths=lengths,
         example_mat=example_mat)
    Slengths,Ss  = gtrd.recover_specs(phn_features,example_mat)
    Ss = Ss.astype(np.float32)
    np.savez('data/Ss_lengths.npz' ,Ss=Ss,Slengths=Slengths,example_mat=example_mat)
    Elengths,Es  = gtrd.recover_edgemaps(phn_features,example_mat,bgd=bgd)
    Es = Es.astype(np.uint8)
    np.savez('data/Es_lengths.npz' ,Es=Es,Elengths=Elengths,example_mat=example_mat)
    # the Es are padded from recover_edgemaps
    for num_mix in num_mix_params:
        print num_mix
        if num_mix == 1:
            affinities = np.ones(Es.shape[0])
            templates = (np.mean(Es,0),)
            spec_templates = (np.mean(Ss,0),)
            np.save('data/%d_affinities.npy' % (num_mix),
                    affinities)
            np.save('data/%d_templates.npy' % (num_mix),
                    templates)
            np.save('data/%d_spec_templates.npy' % (num_mix),
                    spec_templates)
        else:
            bem = bm.BernoulliMixture(num_mix,Es)
            bem.run_EM(.000001)
            templates = et.recover_different_length_templates(bem.affinities,
                                                              Es,
                                                              Elengths)
            spec_templates = et.recover_different_length_templates(bem.affinities,
                                                               Ss,
                                                               Slengths)
            np.save('data/%d_affinities.npy' % (num_mix),
                    bem.affinities)
            np.savez('data/%d_templates.npz' % (num_mix),
                    *templates)
            np.savez('data/%d_spec_templates.npz' % (num_mix),
                    *spec_templates)

def perform_phn_train_detection_SVM(phn, num_mix_params,
                                    train_example_lengths,bgd,
                                    train_path):
    FOMS = collections.defaultdict(list)
    for num_mix in num_mix_params:
        outfile = np.load('%d_templates.npz' % num_mix)
        templates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
        del outfile
        detection_array = np.zeros((train_example_lengths.shape[0],
                            train_example_lengths.max() + 2),dtype=np.float32)
        linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd)
        np.savez('data/linear_filter_%d.npy'% num_mix,*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('data/c_%d.npy'%num_mix,*(tuple(lfc[1] for lfc in linear_filters_cs)))
        syllable = np.array((phn,))
        detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores_mixture(train_path,
                                                                                                   detection_array,
                                                                                                   syllable,
                                                                                                   linear_filters_cs,
                                                                                                       verbose=True)
        np.save('data/detection_array_%d.npy' % num_mix,detection_array)
        if num_mix == 2:
            out = open('data/example_start_end_times_aar.pkl','wb')
            cPickle.dump(example_start_end_times,out)
            out.close()
            out = open('data/detection_lengths_aar.pkl','wb')
            cPickle.dump(detection_lengths,out)
            out.close()
        window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
        window_end = -window_start
        max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                                   example_start_end_times,
                                                                   detection_lengths,
                                                                   window_start,
                                                                   window_end)
        np.save('data/max_detect_vals_%d.npy' % num_mix,max_detect_vals)
        C0 = int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
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
        out = open('data/detection_clusters_%d.npy' % num_mix,
                   'wb')
        cPickle.dump(detection_clusters,out)
        out.close()
        for i in xrange(1,11):
            thresh_idx = np.arange(fpr.shape[0])[fpr*60 <= i].min()
            FOMS[num_mix].append(tpr[thresh_idx])
        template_lengths = tuple(t.shape[0] for t in templates)
        for fnr in first_pass_fnrs:
            thresh_id = int(len(detection_clusters)* fnr/100. + 5)
            (pos_times, 
             false_pos_times, 
             false_neg_times) = rf.get_pos_false_pos_false_neg_detect_points(detection_clusters[thresh_id],
                                                                             detection_array,
                                                                             detection_template_ids,
                                                                             template_lengths,
                                                                             window_start,
                                                                             window_end,example_start_end_times,
                                                                             utterances_path,
                                                                             train_file_indices,
                                                                             verbose=True)
            out = open('data/false_pos_times_%d_%d.pkl' % (num_mix,fnr),'wb')
            pickle.dump(false_pos_times,out)
            out.close()
            out = open('data/pos_times_%d_%d.pkl' % (num_mix,fnr),'wb')
            pickle.dump(pos_times,out)
            out.close()
            out = open('data/false_neg_times_%d_%d.pkl' % (num_mix,fnr),'wb')
            pickle.dump(false_pos_times,out)
            out.close()


