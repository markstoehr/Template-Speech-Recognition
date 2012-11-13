import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import matplotlib.pyplot as plt
import cPickle,os, pickle,collections,itertools

def perform_phn_template_estimation(phn,utterances_path,
                                    file_indices,sp,ep,
                                    num_mix_params,
                                    phn_mapping=None,
                                    waveform_offset=15):
    phn_tuple = (phn,)
    print phn
    phn_features,avg_bgd=gtrd.get_syllable_features_directory(utterances_path,file_indices,phn_tuple,
                                                              S_config=sp,E_config=ep,offset=0,
                                                              E_verbose=False,return_avg_bgd=True,
                                                              waveform_offset=15,
                                                              phn_mapping=phn_mapping)
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
    f = open('data/mixture_estimation_stats_%s.data' % phn,'w')
    for num_mix in num_mix_params:
        print num_mix
        if num_mix == 1:
            affinities = np.ones((Es.shape[0],1),dtype=np.float64)
            mean_length = int(np.mean(Elengths) + .5)
            templates = (np.mean(Es,0)[:mean_length],)
            spec_templates = (np.mean(Ss,0)[:mean_length],)
            np.save('data/%d_affinities.npy' % (num_mix),
                    affinities)
            np.save('data/%d_templates.npy' % (num_mix),
                    templates)
            np.save('data/%d_spec_templates.npy' % (num_mix),
                    spec_templates)
            np.save('data/%d_templates_%s.npy' % (num_mix,phn),
                    templates)
            np.save('data/%d_spec_templates_%s.npy' % (num_mix,phn),
                    spec_templates)
            #
            # write the data to the mixture file for checking purposes
            # format is:
            #   num_components total c0 c1 c2 ... ck
            f.write('%d %d %g\n' % (num_mix,
                                  len(affinities),np.sum(affinities[:,0])))
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
            np.savez('data/%d_templates_%s.npz' % (num_mix,phn),
                    *templates)
            np.savez('data/%d_spec_templates_%s.npz' % (num_mix,phn),
                    *spec_templates)
            f.write('%d %d ' % (num_mix,
                                  len(affinities))
                    + ' '.join(str(np.sum(affinities[:,i]))
                               for i in xrange(affinities.shape[1]))
                               +'\n')
    f.close()



def get_roc_curves(phn, num_mix_params,
                          train_example_lengths,bgd,
                          train_path,file_indices,
                          sp,ep):
    FOMS = collections.defaultdict(list)
    for num_mix in num_mix_params:
        if num_mix > 1:
            outfile = np.load('data/%d_templates.npz' % num_mix)
            templates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
        else:
            templates = (np.load('data/1_templates.npy')[0],)
        detection_array = np.zeros((train_example_lengths.shape[0],
                            train_example_lengths.max() + 2),dtype=np.float32)
        linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd)
        np.savez('data/linear_filter_%d.npz'% num_mix,*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('data/c_%d.npz'%num_mix,*(tuple(lfc[1] for lfc in linear_filters_cs)))
        syllable = np.array((phn,))
        (detection_array,
         example_start_end_times,
         detection_lengths,
         detection_template_ids)=gtrd.get_detection_scores_mixture_named_params(
             train_path,
             file_indices,
             detection_array,
             syllable,
             linear_filters_cs,S_config=sp,
             E_config=ep,
             verbose = True,
             num_examples =-1,
             return_detection_template_ids=True)

        np.save('data/detection_array_%d.npy' % num_mix,detection_array)
        np.save('data/detection_template_ids_%d.npy' % num_mix,detection_template_ids)
        np.save('data/detection_lengths_%d.npy' % num_mix,detection_lengths)
        if num_mix == num_mix_params[0]:
            out = open('data/example_start_end_times.pkl','wb')
            cPickle.dump(example_start_end_times,out)
            out.close()
        window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
        window_end = -window_start
        max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                                   example_start_end_times,
                                                                   detection_lengths,
                                                                   window_start,
                                                                   window_end)
        max_detect_vals = max_detect_vals[:1000]
        np.save('data/max_detect_vals_%d_%s.npy' % (num_mix,phn),max_detect_vals)
        C0 = int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
        C1 = int( 33 * 1.5 + .5)
        frame_rate = 1/.005
        fpr, tpr = rf.get_roc_curve(max_detect_vals,
                                    detection_array,
                                    np.array(detection_lengths),
                                    example_start_end_times,
                                    C0,C1,frame_rate)
        np.save('data/fpr_%d_%s.npy' % (num_mix,phn),
                fpr)
        np.save('data/tpr_%d_%s.npy' % (num_mix,phn),
                tpr)


def get_train_set_division(candidate_thresholds,num_mix,
                           detection_array,detection_lengths,
                           templates,fpr,tpr,first_pass_fnrs):
        detection_clusters = rf.get_detect_clusters_threshold_array(candidate_thresholds,
                                                                    detection_array,
                                                                   detection_lengths,
                                                                    C0,C1)
        out = open('data/detection_clusters_%d.npy' % num_mix,
                   'wb')
        cPickle.dump(detection_clusters,out)
        out.close()
        template_lengths = tuple(t.shape[0] for t in templates)
        for fnr in first_pass_fnrs:
            print "num_mix=%d,fnr=%d" % (num_mix,fnr)
            thresh_id = int(len(detection_clusters)* fnr/100. + 5)
            (pos_times,
             false_pos_times,
             false_neg_times) = rf.get_pos_false_pos_false_neg_detect_points(
                 detection_clusters[thresh_id],
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
            
