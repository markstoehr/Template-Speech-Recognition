#!/usr/bin/python
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

import pickle,collections,cPickle,os,itertools
import argparse
import multiprocessing

SVMResult = collections.namedtuple('SVMResult',
                                   ('num_mix'
                                    +' mix_component'
                                    +' C'
                                    +' W'
                                    +' b'
                                    +' roc_curve'
                                    +' total_error_rate'))
                                    


def get_params(sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False,
               block_length=40,
                            spread_length=1,
                            threshold=.7):
    # we get the basic file paths right here
    # TODO: make this system adaptive
    root_path = '/home/mark/Template-Speech-Recognition/'
    utterances_path = '/home/mark/Template-Speech-Recognition/Data/Train/'
    if os.path.exists('data/train_file_indices.npy'):
        file_indices = np.load('data/train_file_indices.npy')
    else:
        file_indices = gtrd.get_data_files_indices(utterances_path)
        np.save('data/train_file_indices.npy',file_indices)

    num_mix_params = [1,2,3,5,7,9]

    train_file_indices=file_indices
    test_path = '/home/mark/Template-Speech-Recognition/Data/Test/'
    train_path = '/home/mark/Template-Speech-Recognition/Data/Train/'

    if os.path.exists('data/test_file_indices.npy'):
        test_file_indices = np.load('data/test_file_indices.npy')
    else:
        test_file_indices = gtrd.get_data_files_indices(test_path)
        np.save('data/test_file_indices.npy',test_file_indices)

    #file_indices=test_file_indices

    if os.path.exists('data/test_example_lengths.npy'):
        test_example_lengths =np.load("data/test_example_lengths.npy")
    else:
        test_example_lengths = gtrd.get_detect_lengths(test_file_indices,test_path)
        np.save("data/test_example_lengths.npy",test_example_lengths)
        
    if os.path.exists('data/train_example_lengths.npy'):
        train_example_lengths =np.load("data/train_example_lengths.npy")
    else:
        train_example_lengths = gtrd.get_detect_lengths(train_file_indices,train_path)
        np.save("data/train_example_lengths.npy",train_example_lengths)


    return (gtrd.SpectrogramParameters(
            sample_rate=16000,
            num_window_samples=320,
            num_window_step_samples=80,
            fft_length=512,
            kernel_length=7,
            freq_cutoff=3000,
            use_mel=False),
            gtrd.EdgemapParameters(block_length=40,
                                        spread_length=1,
                                        threshold=.7),
            root_path,utterances_path,
            file_indices,num_mix_params,
            test_path,train_path,
            train_example_lengths, train_file_indices,
            test_example_lengths, test_file_indices)


#
# need to access the files where we perform the estimation
#

def get_leehon_mapping():
    """
    Output:
    =======
    leehon_mapping: dict
        dictionary that maps he 62 timit phonetic signals to a more tractable 38
    use_phns: list
        these are the 38 phonetic symbols we map to from the 62 phones
    """
    phns = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 'sp', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'sil']
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
    phns_down = list(
        set(
            [
                p
                for p in phns if p not in leehon_mapping_base.keys() and p != 'q'] + [leehon_mapping_base[p] for p in phns if p in leehon_mapping_base.keys() and p != 'q']))
    leehon_mapping = {}
    for k,v in leehon_mapping_base.items():
        leehon_mapping[k] = v

    for phn in phns_down:
        if phn not in leehon_mapping.keys():
            leehon_mapping[phn] = phn
            
    leehon_mapping["q"] = "q"

    use_phns = np.array(list(set(leehon_mapping.values())))
    return leehon_mapping, use_phns



def save_syllable_features_to_data_dir(phn_tuple,
                          utterances_path,
                          file_indices,
                          
                         sp,ep,
                          phn_mapping,tag_data_with_syllable_string=False,
                                       save_tag="train",
                          waveform_offset=10,
                                       block_features=False,
                                       savedir='data/'):
    """
    Wrapper function to get all the examples processed
    """
    print "Collecting the data for phn_tuple " + ' '.join('%s' % k for k in phn_tuple)
    syllable_string = '_'.join(p for p in phn_tuple)

    phn_features,avg_bgd=gtrd.get_syllable_features_directory(
        utterances_path,
        file_indices,
        phn_tuple,
        S_config=sp,E_config=ep,offset=0,
        E_verbose=False,return_avg_bgd=True,
        waveform_offset=15,
        phn_mapping=phn_mapping)
    bgd = np.clip(avg_bgd.E,.01,.4)
    np.save('data/bgd.npy',bgd)
    example_mat = gtrd.recover_example_map(phn_features)
    lengths,waveforms  = gtrd.recover_waveforms(phn_features,example_mat)
    if tag_data_with_syllable_string:
        np.savez('%s%s_waveforms_lengths_%s.npz' % (savedir,
                                                    syllable_string,
                                                       save_tag),
                 waveforms=waveforms,
                 lengths=lengths,
                 example_mat=example_mat)
    else:
        np.savez('%swaveforms_lengths_%s.npz' % (savedir,
                                                 save_tag),waveforms=waveforms,
                 lengths=lengths,
                 example_mat=example_mat)
    Slengths,Ss  = gtrd.recover_specs(phn_features,example_mat)
    Ss = Ss.astype(np.float32)
    if tag_data_with_syllable_string:
        np.savez('%s%s_Ss_lengths_%s.npz' % (savedir,
                                             syllable_string,
                                                       save_tag),Ss=Ss,Slengths=Slengths,example_mat=example_mat)
    else:
        np.savez('%sSs_lengths_%s.npz' % (
                savedir,
                                             save_tag),Ss=Ss,Slengths=Slengths,example_mat=example_mat)
    Elengths,Es  = gtrd.recover_edgemaps(phn_features,example_mat,bgd=bgd)
    Es = Es.astype(np.uint8)
    if tag_data_with_syllable_string:
        np.savez('%s%s_Es_lengths_%s.npz'% (savedir,
                                             syllable_string,
                                                       save_tag) ,Es=Es,Elengths=Elengths,example_mat=example_mat)
    else:
        np.savez('%sEs_lengths_%s.npz'% (
                savedir,
                save_tag) ,Es=Es,Elengths=Elengths,example_mat=example_mat)

def get_processed_examples(phn_tuple,
                          utterances_path,
                          file_indices,
                          sp,ep,
                          phn_mapping,save_tag="train",
                          waveform_offset=15,
                           return_waveforms=False,
                           savedir='data/'):
    """
    Attempt to load the saved temporary data, if that fails
    then generate it anew
    will later add soemthing for returning the waveforms as needed
    """
    # this hack makes sure that 
    # we only go through the loop below twice at most
    k =0
    while k<2:
        try:
            outfile = np.load('%sEs_lengths_%s.npz' % (savedir,
                                                       save_tag))
            Es = outfile['Es']
            Elengths = outfile['Elengths']
            outfile = np.load('%sSs_lengths_%s.npz' % (savedir,
                                                       save_tag))
            Ss = outfile['Ss']
            Slengths = outfile['Slengths']
            k =2
            return Ss, Slengths, Es,Elengths
        except:
            save_syllable_features_to_data_dir(phn_tuple,
                                               utterances_path,
                                               file_indices,
                                               phn_tuple,
                                               sp,ep,
                                               phn_mapping,
                                               waveform_offset=15)
            outfile = np.load('%sEs_lengths_%s.npz' % (savedir,
                                                       save_tag))
            Es = outfile['Es']
            Elengths = outfile['Elengths']
            outfile = np.load('%sSs_lengths_%s.npz' % (savedir,
                                                       save_tag))
            Ss = outfile['Ss']
            Slengths = outfile['Slengths']
            k+= 1

        if k >2:
            return None, None,None,None


def estimate_templates(num_mix_params,
                       Es,Elengths,
                       Ss,Slengths,
                       get_plots=False,save_tag='',
                       savedir='data/'):
    f = open('%smixture_estimation_stats_regular.data' % savedir ,'w')
    for num_mix in num_mix_params:
        print num_mix
        if num_mix == 1:
            affinities = np.ones((Es.shape[0],1),dtype=np.float64)
            mean_length = int(np.mean(Elengths) + .5)
            templates = (np.mean(Es,0)[:mean_length],)
            spec_templates = (np.mean(Ss,0)[:mean_length],)
            np.save('%s%d_affinities_%s.npy' % (savedir,num_mix,save_tag),
                    affinities)
            np.save('%s%d_templates_%s.npy' % (savedir,num_mix,save_tag),
                    templates)
            np.save('%s%d_spec_templates_%s.npy' % (savedir,num_mix,save_tag),
                    spec_templates)
            np.save('%s%d_templates_%s.npy' % (savedir,num_mix,save_tag),
                    templates)
            np.save('%s%d_spec_templates_%s.npy' % (savedir,num_mix,save_tag),
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
            if get_plots:
                plt.close('all')
                for i in xrange(num_mix):
                    plt.figure()
                    plt.imshow(spec_templates[i].T,origin="lower left")
                    plt.savefig('%s%d_spec_templates_%d_%s.png' % (savedir,num_mix,i,save_tag))
                    plt.close()
                    
            np.save('%s%d_affinities_%s.npy' % (savedir,num_mix,save_tag),
                    bem.affinities)
            np.savez('%s%d_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *templates)
            np.savez('%s%d_spec_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *spec_templates)
            np.savez('%s%d_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *templates)
            np.savez('%s%d_spec_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *spec_templates)
            f.write('%d %d ' % (num_mix,
                                len(bem.affinities))
                + ' '.join(str(np.sum(np.argmax(bem.affinities,1)==i))
                           for i in xrange(bem.affinities.shape[1]))
                    +'\n')
    f.close()


def get_templates(num_mix,template_tag=None,savedir='data/'):
    if template_tag is None:
        if num_mix > 1:
            outfile = np.load('%s%d_templates_regular.npz' % (savedir,num_mix))
            templates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
        else:
            templates = (np.load('%s1_templates_regular.npy' % savedir)[0],)
    else:
        if num_mix > 1:
            outfile = np.load('%s%d_templates_%s.npz' % (savedir,num_mix,template_tag))
            templates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
        else:
            templates = (np.load('%s1_templates_%s.npy' % (savedir,template_tag))[0] ,)
    
    return templates


def compute_slow_detection_scores(num_mix,data_path,file_indices,
                                  sp,ep,leehon_mapping,get_plots=True,
                                  num_utts=-1):
    lf_outfile = np.load('%slinear_filter_%d.npz' % num_mix)
    c_outfile = np.load('%sc_%d.npz' % num_mix)
    linear_filters_cs = zip((tuple(lf_outfile['arr_%d'%i] for i in xrange(num_mix))),
                            (tuple(c_outfile['arr_%d'%i] for i in xrange(num_mix))))
    if num_utts < 1:
        num_utts =len(file_indices)
    for fl_id, fl in enumerate(file_indices[:num_utts]):
        utterance = gtrd.makeUtterance(data_path,fl)
        S = gtrd.get_spectrogram(utterance.s,sp)
        E = gtrd.get_edge_features(S.T,ep,verbose=False
                                       )
        


def save_detection_setup(num_mix,train_example_lengths,
                         train_path,file_indices,syllable,sp,
                         ep,leehon_mapping,
                         save_tag='',template_tag=None,savedir='data/',verbose=False):
    bgd = np.load('%sbgd.npy' %savedir)
    templates =get_templates(num_mix,template_tag=template_tag)
    detection_array = np.zeros((train_example_lengths.shape[0],
                            train_example_lengths.max() + 2),dtype=np.float32)
    linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd)
    if template_tag is None:
        np.savez('%slinear_filter_%d.npz'% (savedir,num_mix),*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('%sc_%d.npz'%(savedir,num_mix),*(tuple(lfc[1] for lfc in linear_filters_cs)))
    else:
        np.savez('%slinear_filter_%d_%s.npz'% (savedir,num_mix,template_tag),*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('%sc_%d_%s.npz'%(savedir,num_mix,template_tag),*(tuple(lfc[1] for lfc in linear_filters_cs)))


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
             verbose = verbose,
             num_examples =-1,
             return_detection_template_ids=True,
             phn_mapping=leehon_mapping )
    np.save('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_array)
    np.save('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_template_ids)
    np.save('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_lengths)
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'wb')
    pickle.dump(example_start_end_times,out)
    out.close()

def get_fpr_tpr(num_mix,
                return_detected_examples=False,
                return_clusters=False):
    detection_array = np.load('%sdetection_array_%d.npy' % num_mix)
    detection_lengths = np.load('%sdetection_lengths_%d.npy' % num_mix)
    out = open('%sexample_start_end_times.pkl','rb')
    example_start_end_times = pickle.load(out)
    out.close()

    templates=get_templates(num_mix)
    window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
    window_end = -window_start
    max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                                   example_start_end_times,
                                                                   detection_lengths,
                                                                   window_start,
                                                                   window_end)
    np.save('%smax_detect_vals_%d.npy' % (num_mix),max_detect_vals)
    C0 = int(np.max(tuple( t.shape[0] for t in templates))+.5)
    C1 = int( 33 * 1.5 + .5)
    frame_rate = 1/.005
    roc_out  = rf.get_roc_curve(max_detect_vals,
                                detection_array,
                                detection_lengths,
                                example_start_end_times,
                                C0,C1,frame_rate,
                                return_detected_examples=return_detected_examples,
                                return_clusters=return_clusters)
    np.save('%sfpr_%d.npy' % (num_mix),
            roc_out[0])
    np.save('%stpr_%d.npy' % (num_mix),
                roc_out[1])
    if len(roc_out) > 2:
        if len(roc_out)==4:
            np.save('%sdetected_examples.npy',roc_out[2])
            np.save('%sroc_clusters.npy',roc_out[3])
        elif return_clusters:
            np.save('%sroc_clusters.npy',roc_out[2])
        else:
            np.save('%sdetected_examples.npy',roc_out[2])


def get_fpr_tpr_tagged(num_mix,syllable_string,
                       return_detected_examples=False,
                       return_clusters=False,
                       save_tag='',savedir='data/',
                       get_plots=False):
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'rb')
    example_start_end_times = pickle.load(out)
    out.close()

    templates=get_templates(num_mix)
    window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
    window_end = -window_start
    max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                                   example_start_end_times,
                                                                   detection_lengths,
                                                                   window_start,
                                                                   window_end)
    np.save('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_vals)
    C0 = int(np.max(tuple( t.shape[0] for t in templates))+.5)
    C1 = int( 33 * 1.5 + .5)
    frame_rate = 1/.005
    roc_out  = rf.get_roc_curve(max_detect_vals,
                                detection_array,
                                detection_lengths,
                                example_start_end_times,
                                C0,C1,frame_rate,
                                return_detected_examples=return_detected_examples,
                                return_clusters=return_clusters)
    np.save('%sfpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
            roc_out[0])
    roc_out[1][:] = np.clip(roc_out[1],0.,1.)
    np.save('%stpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
                roc_out[1])
    if get_plots:
        plt.close('all')
        plt.plot(roc_out[1],roc_out[0])
        plt.xlabel('Percent True Positives Retained')
        plt.ylabel('False Positives Per Second')
        plt.title('ROC %s 1-stage Likelihood num_mix=%d' %(syllable_string,
                                                                  num_mix))
        plt.savefig('%s%s_fp_roc_discriminationLike_1stage_%d_%s.png' % (savedir,syllable_string,
                                                     
                                                            num_mix,
                                                                          save_tag))
        plt.close('all')
    if len(roc_out) > 2:
        if len(roc_out)==4:
            np.save('%sdetected_examples.npy',roc_out[2])
            np.save('%sroc_clusters.npy',roc_out[3])
        elif return_clusters:
            np.save('%sroc_clusters.npy',roc_out[2])
        else:
            np.save('%sdetected_examples.npy',roc_out[2])


def get_estimated_detection_clusters(num_mix):
    detection_array = np.load('%sdetection_array_%d.npy' % num_mix)    
    max_detect_vals = np.load('%smax_detect_vals_%d.npy' % (num_mix))
    detection_lengths = np.load('%sdetection_lengths_%d.npy' % num_mix)
    templates=get_templates(num_mix)
    C0 = int(np.mean(tuple( t.shape[0] for t in templates))+.5)
    C1 = int( 33 * 1.5 + .5)
    detection_clusters = rf.get_detect_clusters_threshold_array(
        max_detect_vals,
        detection_array,
        detection_lengths,
        C0,C1)
    out = open('%sdetection_clusters.pkl','wb')
    cPickle.dump(detection_clusters,out)
    out.close()
    return detection_clusters

def get_tagged_detection_clusters(num_mix,thresh_percent,save_tag='',use_thresh=None,old_max_detect_tag=None,savedir='data/'):
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))    
    if old_max_detect_tag is None:
        max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,save_tag))
    else:
        max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,old_max_detect_tag))
        
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))
    templates=get_templates(num_mix)
    C0 = int(np.mean(tuple( t.shape[0] for t in templates))+.5)
    C1 = int( 33 * 1.5 + .5)
    thresh_id = int((len(max_detect_vals)-1) *thresh_percent/float(100))
    if use_thresh is None:
        detect_thresh =max_detect_vals[thresh_id]
    else:
        detect_thresh =use_thresh
    detection_clusters = rf._get_detect_clusters_single_threshold(detect_thresh,
                                          detection_array,
                                          detection_lengths,
                                          C0,C1)
    out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (savedir,num_mix,
                                                                       thresh_percent,
                                                                       save_tag)
                                                                    ,'wb')
    cPickle.dump(detection_clusters,out)
    out.close()
    return detection_clusters
    



def get_true_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',savedir='data/',
                           verbose=False):
    if thresh_percent is None and save_tag=='':
        out = open('%s%s_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')
    true_pos_times=pickle.load(out)
    out.close()
    true_pos_scores = np.array(tuple(
        fpd.cluster_max_peak_val
        for fpd in reduce(lambda x,y: x+y,true_pos_times)))
    np.save('%s%s_true_positive_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),
            true_pos_scores)
    true_positives = rf.get_true_positives(true_pos_times,
                                             S_config=sp,
                                             E_config=ep,
                                             offset=0,
                                             waveform_offset=waveform_offset,
                                             verbose=verbose)
    example_mat = gtrd.recover_example_map(true_positives)
    np.save('%s%s_true_positives_example_mat_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),example_mat)
    lengths,waveforms  = gtrd.recover_waveforms(true_positives,example_mat)
    np.savez('%s%s_true_positives_waveforms_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                                   num_mix,
                                                                         thresh_percent,
                                                                         save_tag),
             lengths=lengths,
             waveforms=waveforms)
    Slengths,Ss  = gtrd.recover_specs(true_positives,example_mat)
    np.savez('%s%s_true_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Slengths,
             Ss=Ss)
    Elengths,Es  = gtrd.recover_edgemaps(true_positives,example_mat)
    np.savez('%s%s_true_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Elengths,
             Es=Es)

def get_false_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',savedir='data/',
                           verbose=False):
    if thresh_percent is None and save_tag=='':
        out = open('%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')
    false_pos_times=pickle.load(out)
    out.close()
    false_pos_scores = np.array(tuple(
        fpd.cluster_max_peak_val
        for fpd in reduce(lambda x,y: x+y,false_pos_times)))
    np.save('%s%s_false_positive_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),
            false_pos_scores)
    false_positives = rf.get_false_positives(false_pos_times,
                                             S_config=sp,
                                             E_config=ep,
                                             offset=0,
                                             waveform_offset=waveform_offset,
                                             verbose=verbose)
    example_mat = gtrd.recover_example_map(false_positives)
    np.save('%s%s_false_positives_example_mat_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),example_mat)
    lengths,waveforms  = gtrd.recover_waveforms(false_positives,example_mat)
    np.savez('%s%s_false_positives_waveforms_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                                   num_mix,
                                                                         thresh_percent,
                                                                         save_tag),
             lengths=lengths,
             waveforms=waveforms)
    Slengths,Ss  = gtrd.recover_specs(false_positives,example_mat)
    np.savez('%s%s_false_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Slengths,
             Ss=Ss)
    Elengths,Es  = gtrd.recover_edgemaps(false_positives,example_mat)
    np.savez('%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Elengths,
             Es=Es)

def get_false_neg_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',savedir='data/',
                           verbose=False):
    if thresh_percent is None and save_tag=='':
        out = open('%s%s_false_neg_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')
    false_neg_times=pickle.load(out)
    out.close()
    false_neg_scores = np.array(tuple(
        fpd.max_peak_val
        for fpd in reduce(lambda x,y: x+y,false_neg_times)))
    np.save('%s%s_false_negative_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),
            false_neg_scores)
    false_negatives = rf.get_false_negatives(false_neg_times,
                                             sp,
                                             ep,
                                             offset=0,
                                             waveform_offset=waveform_offset,
                                             verbose=verbose)
    example_mat = gtrd.recover_example_map(false_negatives)
    np.save('%s%s_false_negatives_example_mat_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),example_mat)
    lengths,waveforms  = gtrd.recover_waveforms(false_negatives,example_mat)
    np.savez('%s%s_false_negatives_waveforms_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                                   num_mix,
                                                                         thresh_percent,
                                                                         save_tag),
             lengths=lengths,
             waveforms=waveforms)
    Slengths,Ss  = gtrd.recover_specs(false_negatives,example_mat)
    np.savez('%s%s_false_negatives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Slengths,
             Ss=Ss)
    Elengths,Es  = gtrd.recover_edgemaps(false_negatives,example_mat)
    np.savez('%s%s_false_negatives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Elengths,
             Es=Es)


def get_detection_clusters_by_label(num_mix,utterances_path,
                                    file_indices,thresh_percent,single_threshold=True,save_tag='',verbose=False, savedir='data/',
                                    return_example_types=False,
  
                                    detect_clusters=None):
    if verbose:
        print "save_tag=%s" % save_tag
        
    if detect_clusters is not None:
        if verbose:
            print "Using detect_clusters no file loading"
        detection_clusters=detect_clusters
    else:
        if single_threshold:
            out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (savedir,num_mix,
                                                                       thresh_percent,
                                                                       save_tag),'rb')
        elif save_tag != '':
            out = open('%sdetection_clusters_%s.pkl' %(savedir,save_tag),'rb')
        else:
            out = open('%sdetection_clusters.pkl'%savedir,'rb')
            
        detection_clusters =cPickle.load(out)
        out.close()

    if save_tag == '':
        detection_template_ids = np.load('%sdetection_template_ids_%d.npy' % (savedir,num_mix))
        detection_array = np.load('%sdetection_array_%d.npy' % (savedir,num_mix))
        detection_lengths = np.load('%sdetection_lengths_%d.npy' % (savedir,num_mix))
        out = open('%sexample_start_end_times.pkl' %savedir,'rb')
    else:
        detection_template_ids = np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,save_tag))
        detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))
        detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                          save_tag))
        out = open('%sexample_start_end_times_%s.pkl' %(savedir,save_tag),'rb')

    example_start_end_times = pickle.load(out)
    out.close()

    templates=get_templates(num_mix)
    template_lengths = np.array([len(t) for t in templates])
    
    window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
    window_end = -window_start
    
    if single_threshold: 
        return rf.get_pos_false_pos_false_neg_detect_points(detection_clusters,
                                              detection_array,
                                              detection_template_ids,
                                              template_lengths,
                                              window_start,
                                              window_end,example_start_end_times,
                                              utterances_path,
                                              file_indices,
                                              verbose=verbose,
                                                        return_example_types=return_example_types)
    else:
        detection_cluster_idx = int(len(detection_clusters) * thresh_percent/100.)
    
        return rf.get_pos_false_pos_false_neg_detect_points(detection_clusters[detection_cluster_idx],
                                              detection_array,
                                              detection_template_ids,
                                              template_lengths,
                                              window_start,
                                              window_end,example_start_end_times,
                                              utterances_path,
                                              file_indices,
                                              verbose=verbose,
                                                        return_example_types=return_example_types) +(detection_cluster_idx,)

def perform_second_stage_detection_testing(num_mix,syllable_string,save_tag,thresh_percent,
                                           make_plots=False,verbose=False):
    out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (syllable_string,num_mix,thresh_percent,save_tag),'rb')
    false_pos_times=pickle.load(out)
    out.close()
    template_ids = rf.recover_template_ids_detect_times(false_pos_times)
    outfile = np.load('%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (syllable_string,num_mix,thresh_percent,save_tag))
    lengths_false_pos = outfile['lengths']
    Es_false_pos = outfile['Es']
    outfile = np.load('%s%s_false_positives_Ss_lengths_%d_%d_%s.npz' % (syllable_string,num_mix,thresh_percent,save_tag))
    lengths_S_false_pos = outfile['lengths']
    Ss_false_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts
    templates = get_templates(num_mix)
    Es_false_pos_clusters = rf.get_false_pos_clusters(Es_false_pos,
                                               templates,
                                               template_ids)
    Ss_false_pos_clusters =rf.get_false_pos_clusters(Ss_false_pos,
                                               templates,
                                               template_ids)

    false_pos_cluster_counts = np.array([len(k) for k in Es_false_pos_clusters])
    if verbose:
        for false_pos_idx, false_pos_count in enumerate(false_pos_cluster_counts):
            print "Template %d had %d false positives" %(false_pos_idx,false_pos_count)
    out = open('%s%s_pos_times_%d_%d_%s.pkl' % (syllable_string,num_mix,thresh_percent,save_tag),'rb')
    true_pos_times=pickle.load(out)
    out.close()
    template_ids = rf.recover_template_ids_detect_times(true_pos_times)

    outfile = np.load('%s%s_true_positives_Es_lengths_%d_%d_%s.npz' % (syllable_string,num_mix,thresh_percent,save_tag))
    lengths_true_pos = outfile['lengths']
    Es_true_pos = outfile['Es']
    outfile = np.load('%s%s_true_positives_Ss_lengths_%d_%d_%s.npz' % (syllable_string,num_mix,thresh_percent,save_tag))
    lengths_S_true_pos = outfile['lengths']
    Ss_true_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts
    templates = get_templates(num_mix)
    Es_true_pos_clusters = rf.get_false_pos_clusters(Es_true_pos,
                                               templates,
                                               template_ids)
    Ss_true_pos_clusters =rf.get_false_pos_clusters(Ss_true_pos,
                                               templates,
                                               template_ids)

    true_pos_cluster_counts = np.array([len(k) for k in Es_true_pos_clusters])
    if verbose:
        for true_pos_idx, true_pos_count in enumerate(true_pos_cluster_counts):
            print "Template %d had %d true positives" %(true_pos_idx,true_pos_count)
    clusters_for_classification = tuple(
        np.vstack((Es_true_cluster,Es_false_cluster))
        for Es_true_cluster, Es_false_cluster in itertools.izip(
            Es_true_pos_clusters,
            Es_false_pos_clusters))
    np.savez('%s%s_clusters_for_testing_%d_%d_%s.npz' % (syllable_string,num_mix,thresh_percent,save_tag),
             *clusters_for_classification)

    labels_for_classification = tuple(
        np.hstack((np.ones(Es_true_cluster.shape[0]),
                   np.zeros(Es_false_cluster.shape[0])))
        for Es_true_cluster, Es_false_cluster in itertools.izip(
            Es_true_pos_clusters,
            Es_false_pos_clusters))

    np.savez('%s%s_labels_for_testing_%d_%d_%s.npz' % (syllable_string,num_mix,thresh_percent,save_tag),
             *labels_for_classification)

def get_roc_curve(detector_neg_scores,detector_pos_scores,
                  num_mix,thresh_percent,syllable_string,
                  save_tag,classifier_name=None,plot_name=None):
    false_neg_scores = np.load('%s%s_false_negative_scores_%d_%d_%s.npy' % (syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))

    false_pos_scores = np.load('%s%s_false_positive_scores_%d_%d_%s.npy' % (syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))
    true_pos_scores = np.load('%s%s_true_positive_scores_%d_%d_%s.npy' % (syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))

    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (num_mix,
                                                save_tag))
    time_detecting = .005 * detection_lengths.sum()
    num_true = float(len(true_pos_scores)
                               + len(false_neg_scores))

    sorted_scores = np.sort(true_pos_scores)
    baseline_fpr = np.array([
            np.sum(false_pos_scores > sorted_scores[k])
            for k in xrange(sorted_scores.shape[0])])/ time_detecting
    tpr = np.arange(len(true_pos_scores),dtype=np.float)/num_true

    classifier_fpr = np.array([
            np.sum(detector_neg_scores>detector_pos_scores[k])
            for k in xrange(detector_pos_scores.shape[0])])/time_detecting

    if plot_name is not None:
        plt.close('all')
        plt.plot(base_fpr,tpr,c='r',label='baseline')
        plt.plot(classifier_fpr,tpr,c='b',label=classifier_name)
        plt.legend('baseline',classifier_name)
        plt.xlabel('Percent True Positives Retained')
        plt.ylabel('Percent False Positives Retained')
        plt.title('ROC %s %s num_mix=%d mix_id=%d' %(syllable_string,plot_name,
                                                                  num_mix,
                                                                  mix_component))
        plt.savefig('%s_%s_%d_%d_%s.png' % (syllable_string,plot_name,
                                            mix_component,
                                            num_mix,save_tag))
        plt.close('all')



def run_all_linear_filters(num_mix,syllable_string,save_tag,thresh_percent,make_plots=False,savedir='data/',verbose=False):
    outfile =np.load('%s%s_clusters_for_testing_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
             )
    clusters_for_classification = tuple(
        outfile['arr_%d' % k] for k in xrange(num_mix))
    outfile =np.load('%s%s_labels_for_testing_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
             )
    labels_for_classification = tuple(
        outfile['arr_%d' % k] for k in xrange(num_mix))
                           
    outfile = np.load('%s%s_lf_c_second_stage_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,save_tag))
    lf_second_stage = outfile['lf']
    c_second_stage = outfile['c']
    
    outfile = np.load('%s%s_lf_c_second_stage_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,save_tag))

    false_neg_scores = np.load('%s%s_false_negative_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))

    false_pos_scores = np.load('%s%s_false_positive_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))
    true_pos_scores = np.load('%s%s_true_positive_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))
    
                              
                               
    
    

def run_fp_detector(num_mix,syllable_string,new_tag,thresh_percent=None,save_tag=None,make_plots=False,savedir='data/',
                    verbose=False):
    if thresh_percent is None:
        out = open('%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),'rb')
    false_pos_times=pickle.load(out)
    out.close()
    template_ids = rf.recover_template_ids_detect_times(false_pos_times)
    outfile = np.load('%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_false_pos = outfile['lengths']
    Es_false_pos = outfile['Es']
    outfile = np.load('%s%s_false_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_S_false_pos = outfile['lengths']
    Ss_false_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts
    templates = get_templates(num_mix)
    Es_false_pos_clusters = rf.get_false_pos_clusters(Es_false_pos,
                                               templates,
                                               template_ids)
    Ss_false_pos_clusters =rf.get_false_pos_clusters(Ss_false_pos,
                                               templates,
                                               template_ids)

    false_pos_cluster_counts = np.array([len(k) for k in Es_false_pos_clusters])
    if verbose:
        for false_pos_idx, false_pos_count in enumerate(false_pos_cluster_counts):
            print "Template %d had %d false positives" %(false_pos_idx,false_pos_count)
    bgd =np.load('%sbgd.npy' %savedir)
    Es_false_pos_clusters2 = ()
    Ss_false_pos_clusters2 =()
    for i,fpc in enumerate(Es_false_pos_clusters):
        if fpc.shape[0] == 0:
            rep_bgd =np.tile(bgd.reshape((1,)+bgd.shape),
                                            (500,) + tuple(np.ones(len(bgd.shape))))
            Es_false_pos_clusters2 += ( (rep_bgd > np.random.rand(*(rep_bgd.shape))).astype(np.uint8),)
            Ss_false_pos_clusters2 += (np.zeros((500,) +Ss_false_pos_clusters[i].shape[1:]),)
        else:
            Es_false_pos_clusters2 += (Es_false_pos_clusters[i],)
            Ss_false_pos_clusters2 += (Ss_false_pos_clusters[i],)

    Es_false_pos_clusters = Es_false_pos_clusters2
    Ss_false_pos_clusters = Ss_false_pos_clusters2

    np.savez('%s%s_Es_false_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Es_false_pos_clusters)
    np.savez('%s%s_Ss_false_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Ss_false_pos_clusters)
    template_affinities = np.load('%s%d_affinities_regular.npy' % (savedir,num_mix))
    outfile = np.load('%sEs_lengths_%s.npz' % (savedir,save_tag))
    Es_true_pos = outfile['Es']
    Elengths_true_pos = outfile['Elengths']
    outfile = np.load('%sSs_lengths_%s.npz' % (savedir,save_tag))
    Ss_true_pos = outfile['Ss']
    Slengths_true_pos = outfile['Slengths']
    clustered_training_true_Es = et.recover_clustered_data(template_affinities,
                                                           Es_true_pos,
                                                           templates,
                                                           assignment_threshold = .95)
    clustered_training_true_Ss = et.recover_clustered_data(template_affinities,
                                                           Ss_true_pos,
                                                           templates,
                                                           assignment_threshold = .95)
    np.savez('%s%s_training_true_pos_Es_%d_%s.npz' %(savedir,syllable_string,num_mix,new_tag), *clustered_training_true_Es)
    np.savez('%s%s_training_true_pos_Ss_%d_%s.npz' %(savedir,syllable_string,num_mix,new_tag), *clustered_training_true_Ss)
    # learn a template on half the false positive data
    # do for each mixture component and see the curve
    # need to compute the likelihood ratio test
    # outfile = np.load('%s%s_training_true_pos_Es_%d_%s.npz' %(syllable_string,num_mix,save_tag))
    # clustered_training_true_Es = tuple(outfile['arr_%d' %i] for i in xrange(len(outfile.files)))
    # outfile = np.load('%s%s_training_true_pos_Ss_%d_%s.npz' %(syllable_string,num_mix,save_tag))
    # clustered_training_true_Ss = tuple(outfile['arr_%d' %i] for i in xrange(len(outfile.files)))

    # outfile = np.load('%s%s_Es_false_pos_clusters_%d.npz' %(syllable_string,num_mix))
    # Es_false_pos_clusters = tuple(outfile['arr_%d' %i] for i in xrange(len(outfile.files)))
    # outfile = np.load('%s%s_Ss_false_pos_clusters_%d.npz' %(syllable_string,num_mix))
    # Ss_false_pos_clusters = tuple(outfile['arr_%d' %i] for i in xrange(len(outfile.files)))
    for mix_component in xrange(num_mix):
        if verbose:
            print "Working on mixture component: %d" % mix_component
        if make_plots:
            plt.close('all')
            plt.imshow(np.mean(Ss_false_pos_clusters[mix_component],0).T[::-1],
                       origin="lower left")
            plt.savefig('%s%s_Ss_false_pos_template_%d_%d_%s.png' % (savedir,syllable_string,num_mix,mix_component,new_tag))
            plt.close('all')
        get_baseline_second_stage_detection(clustered_training_true_Es[mix_component],Es_false_pos_clusters[mix_component],
                                            templates[mix_component], num_mix,mix_component,
                                            syllable_string,new_tag,savedir,
                                            make_plots=make_plots)
        get_svm_second_stage_detection(clustered_training_true_Es[mix_component],Es_false_pos_clusters[mix_component],
                                            templates[mix_component], num_mix,mix_component,
                                            syllable_string,new_tag,savedir,
                                            make_plots=make_plots)

def get_baseline_second_stage_detection(true_pos_cluster,false_pos_cluster,
                                        template, num_mix,mix_component,
                                        syllable_string,save_tag,savedir='data/',
                                        make_plots=False):
    num_false_pos_component =false_pos_cluster.shape[0]
    false_pos_template = np.clip(np.mean(false_pos_cluster[:num_false_pos_component/2],0),.01,.99)
    lf,c = et.construct_linear_filter_structured_alternative(
        template,
        false_pos_template,
        
        bgd=None,min_prob=.01)
    np.savez('%s%s_lf_c_second_stage_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,save_tag),
             lf=lf,
             c=c)
    true_responses = np.sort(np.sum(true_pos_cluster * lf + c,-1).sum(-1).sum(-1))
    false_responses = np.sort((false_pos_cluster[num_false_pos_component/2:]*lf+ c).sum(-1).sum(-1).sum(-1))
    roc_curve = np.array([
                np.sum(false_responses >= true_response)/float(len(false_responses))
                for true_response in true_responses]) 
    np.save('%s%s_fpr_detector_rocLike_%d_%d_%s.npy' % (syllable_string,
                                                     mix_component,
                                                     num_mix,save_tag),roc_curve)
    if make_plots:
        plt.close('all')
        plt.plot(1-np.arange(roc_curve.shape[0],dtype=float)/roc_curve.shape[0],roc_curve)
        plt.xlabel('Percent True Positives Retained')
        plt.ylabel('Percent False Positives Retained')
        plt.title('ROC %s Likelihood num_mix=%d mix_id=%d' %(syllable_string,
                                                                  num_mix,
                                                                  mix_component))
        plt.savefig('%s%s_fp_roc_discriminationLike_%d_%d_%s.png' % (savedir,syllable_string,
                                                     mix_component,
                                                            num_mix,save_tag))
        plt.close('all')

def get_svm_second_stage_detection(true_pos_cluster,false_pos_cluster,
                                        template, num_mix,mix_component,
                                        syllable_string,save_tag,savedir='data/',
                                   penalty_list=(('unreg', 1), 
                                                 ('little_reg',.1), 
                                                 ('reg', 0.05),
                                                 ('reg_plus', 0.01),
                                                 ('reg_plus_plus',.001)),
                                        make_plots=False):
    data_shape = true_pos_cluster[0].shape
    num_true = len(true_pos_cluster)
    num_false = len(false_pos_cluster)
    num_true_train = int(num_true * .75)
    num_false_train = int(num_false * .5)
    training_data_X = np.vstack((
            true_pos_cluster[:num_true_train].reshape(
                num_true_train,
                np.prod(data_shape)),
            false_pos_cluster[:num_false_train].reshape(
                num_false_train,
                np.prod(data_shape))))
    training_data_Y = np.hstack((
            np.ones(num_true_train),
            np.zeros(num_false_train)))
    testing_data_X =  np.vstack((
            true_pos_cluster[num_true_train:].reshape(
                num_true - num_true_train,
                np.prod(data_shape)),
            false_pos_cluster[num_false_train:].reshape(
                num_false - num_false_train,
                np.prod(data_shape))))
    testing_data_Y = np.hstack((
                np.ones(num_true-num_true_train),
                np.zeros(num_false-num_false_train)))

    for name, penalty in penalty_list:
        clf = svm.SVC(kernel='linear', C=penalty)
        clf.fit(training_data_X, training_data_Y)
        # get the roc curve
        w = clf.coef_[0]
        b = clf.intercept_[0]
        np.savez('%s%s_w_b_second_stage_%d_%d_%s_%s.npz' % (savedir,syllable_string,
                                                          mix_component,
                                                          num_mix,
                                                               name,save_tag),
                 w=w,
                 b=b)
        testing_raw_outs = (testing_data_X * w + b).sum(1)
        val_thresholds = np.sort(testing_raw_outs[testing_data_Y==1])
        roc_curve = np.zeros(len(val_thresholds))
        num_neg = float(np.sum(testing_data_Y==0))
        for i,thresh in enumerate(val_thresholds):
            roc_curve[i] = np.sum(testing_raw_outs[testing_data_Y==0]  <thresh)/num_neg
        np.save('%s%s_fpr_detector_rocSVM_%d_%d_%s_%s.npy' % (savedir,syllable_string,
                                                              mix_component,
                                                              num_mix,
                                                              name,save_tag),roc_curve)
        if make_plots:
            plt.close('all')
            plt.plot(1-np.arange(roc_curve.shape[0])/float(roc_curve.shape[0]),1-roc_curve)
            plt.xlabel('Percent True Positives Retained')
            plt.ylabel('Percent False Positives Retained')
            plt.title('ROC %s SVM penalty=%g num_mix=%d mix_id=%d' %(syllable_string,penalty,
                                                                  num_mix,
                                                                  mix_component))
            plt.savefig('%sroc_%s_layer_2SVM_%s_%d_%d_%s.png' %(savedir,
                                                                syllable_string,
                                                                name,
                                                                num_mix,
                                                                  mix_component,save_tag))

            plt.close('all')


def get_test_clusters_for_2nd_stage(num_mix,data_path,
                               file_indices,
                               save_tag='',savedir='data/',
                                    verbose=False):
    out = open('%sdetection_clusters_single_thresh_%d_%s.pkl' % (savedir,num_mix,save_tag)                                                                    ,'rb')
    detection_clusters=cPickle.load(out)
    out.close()

    
    # need to find max point of clusters (perhaps this should
    # happen during the cluster identification phase)
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    detection_template_ids=np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'rb')
    example_start_end_times=pickle.load(out)
    out.close()

    templates = tuple( k for k in get_templates(num_mix))
    template_lengths = tuple( len(k) for k in templates)
    window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
    window_end = -window_start

    (pos_times,
     false_pos_times,
     false_neg_times)=rf.get_pos_false_pos_false_neg_detect_points(detection_clusters,
                                                 detection_array,
                                                 detection_template_ids,
                                                 template_lengths,
                                                 window_start,
                                                 window_end,example_start_end_times,
                                                 data_path,
                                                 file_indices,
                                                 verbose=verbose)
    num_pos_detections_by_label

    clustered_Es = np.empty(num_mix,dtype=object)
    clustered_Ss = np.empty(num_mix,dtype=object)
    clustered_labels = np.empty(num_mix,dtype=object)
    clustered_waveforms = np.empty(num_mix,dtype=object)
        
    
    cluster_max_peak_loc = rf.get_max_peak(cluster_vals)

    if make_plots:
            plt.close('all')
            plt.plot(1-np.arange(roc_curve.shape[0])/float(roc_curve.shape[0]),1-roc_curve)
            plt.xlabel('Percent True Positives Retained')
            plt.ylabel('Percent False Positives Retained')
            plt.title('ROC %s SVM penalty=%g num_mix=%d mix_id=%d' %(syllable_string,penalty,
                                                                  num_mix,
                                                                  mix_component))
            plt.savefig('%sroc_%s_layer_2SVM_%s_%d_%d' %(savedir,
                                                         syllable_string,name,
                                                                  num_mix,
                                                                  mix_component))

            plt.close('all')

def main2(syllable,threshval,
         make_plots=False):
    (sp,
     ep,
     root_path,utterances_path,
     file_indices,num_mix_params,
     test_path,train_path,
     train_example_lengths, train_file_indices,
     test_example_lengths, test_file_indices) = get_params()
    (sp2,
     ep2,
     root_path,utterances_path,
     file_indices,num_mix_params,
     test_path,train_path,
     train_example_lengths, train_file_indices,
     test_example_lengths, test_file_indices) = get_params(spread_length=2)
    leehon_mapping, use_phns = get_leehon_mapping()
    leehon_mapping =None
    # leehon_mapping=None
    num_mix_params = [9]
    save_syllable_features_to_data_dir(syllable,
                          train_path,
                          train_file_indices,
                          sp2,ep2,
                          leehon_mapping,save_tag="train_2",
                          waveform_offset=10)
    print "Finished save_syllabel_features_to_data_dir"
    (Ss, 
     Slengths, 
     Es,
     Elengths) =get_processed_examples(syllable,
                           train_path,
                           train_file_indices,
                           sp,ep,
                           leehon_mapping,save_tag='train_2',
                           waveform_offset=10,
                           return_waveforms=False)
    print "Finished get_processed_examples"
    num_mix = 9
    num_mix_params=np.arange(9)+1
    estimate_templates(num_mix_params,
                       Es,Elengths,
                       Ss,Slengths,
                       get_plots=True,save_tag='train_2')
    print "Finished estimate_templates"
    for num_mix in num_mix_params[1:]:
        print num_mix
        save_detection_setup(num_mix,test_example_lengths,
                             test_path,test_file_indices,syllable,sp,
                             ep,leehon_mapping,save_tag='test_2',
                             template_tag='train_2')
        print "Finished save_detection_setup"
    #get_estimated_detection_clusters(num_mix)
        syllable_string = '_'.join(p for p in syllable)
        get_fpr_tpr_tagged(num_mix,syllable_string,
                           return_detected_examples=False,
                           return_clusters=False,
                           save_tag='test_2',
                           get_plots=True)

    thresh_percent =1
    get_tagged_detection_clusters(num_mix,thresh_percent,save_tag='train')

    (pos_times,
     false_pos_times,
     false_neg_times,
     example_types,
     ) = get_detection_clusters_by_label(num_mix,train_path,
                                    train_file_indices,thresh_percent,single_threshold=True,save_tag='train',verbose=False,return_example_types=True)
    syllable_string = '_'.join(p for p in syllable)
    out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (syllable_string,num_mix,
                                                      thresh_percent,save_tag),'wb')
    pickle.dump(false_pos_times,out)
    out.close()
    out = open('%s%s_pos_times_%d_%d_%s.pkl' % (syllable_string,num_mix,
                                                thresh_percent,save_tag),'wb')
    pickle.dump(pos_times,out)
    out.close()
    out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (syllable_string,num_mix,
                                                      thresh_percent,save_tag),'wb')
    pickle.dump(false_neg_times,out)
    out.close()
    # now time to run things on
    get_false_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='train',
                           verbose=False)
    run_fp_detector(num_mix,syllable_string,"koloy_test",make_plots=True,
                    thresh_percent=thresh_percent,save_tag='train',verbose=True)
    print "Starting test-phase save_detection_setup"
    save_detection_setup(num_mix,test_example_lengths,
                         test_path,test_file_indices,syllable,sp,
                         ep,leehon_mapping,save_tag="test")
    # going to use detection at the same threshold
    get_fpr_tpr_tagged(num_mix,syllable_string,
                return_detected_examples=False,
                return_clusters=False,
                save_tag='test',get_plots=True)
    test_detect_clusters= get_tagged_detection_clusters(num_mix,thresh_percent,save_tag='test',old_max_detect_tag='train',use_thresh=0.0)
    (pos_times,
     false_pos_times,
     false_neg_times,
     example_types,
     ) = get_detection_clusters_by_label(num_mix,test_path,
                                    test_file_indices,thresh_percent,single_threshold=True,save_tag='test',verbose=False,return_example_types=True,
                                         detect_clusters=test_detect_clusters)
    out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                      thresh_percent,'test'),'wb')
    pickle.dump(false_pos_times,out)
    out.close()
    out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                thresh_percent,'test'),'wb')
    pickle.dump(pos_times,out)
    out.close()
    out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                      thresh_percent,'test'),'wb')
    pickle.dump(false_neg_times,out)
    out.close()


    get_false_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='test',
                           verbose=True)
    get_true_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='test',
                           verbose=True)
    get_false_neg_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='test',
                           verbose=True)

    perform_second_stage_detection_testing(num_mix,syllable_string,'test',thresh_percent,
                                           make_plots=False,verbose=False)
    
    save_syllable_features_to_data_dir(syllable,
                                       test_path,
                                       test_file_indices,
                          
                                       sp,ep,
                                       leehon_mapping,tag_data_with_syllable_string=False,
                                       save_tag="test",
                                       waveform_offset=10)

def main(args):
    if args.v:
        print args

    syllable_string = '_'.join(p for p in args.detect_object)
    (sp,
     ep,
     root_path,utterances_path,
     file_indices,num_mix_params,
     test_path,train_path,
     train_example_lengths, train_file_indices,
     test_example_lengths, test_file_indices) = get_params(
        sample_rate=args.sample_rate,
        num_window_samples=args.num_window_samples,
        fft_length=args.fft_length,
        kernel_length=args.kernel_length,
        freq_cutoff=args.freq_cutoff,
        use_mel=args.use_mel,
        block_length=args.block_length,
        spread_length=args.spread_length,
        threshold=args.edge_threshold_quantile)
    if args.leehon_mapping:
        leehon_mapping, use_phns = get_leehon_mapping()
    else:
        leehon_mapping =None
    if args.save_syllable_features_to_data_dir != '':
        save_syllable_features_to_data_dir(args.detect_object,
                                           train_path,
                                           train_file_indices,
                                           sp,ep,
                                           leehon_mapping,save_tag=args.save_tag,
                                           waveform_offset=10,
                                           savedir=args.save_syllable_features_to_data_dir)
        print "Finished save_syllable_features_to_data_dir"
        savedir=args.save_syllable_features_to_data_dir
    else:
        savedir=args.savedir
    if args.estimate_templates:
        (Ss, 
         Slengths, 
         Es,
         Elengths) =get_processed_examples(args.detect_object,
                                           train_path,
                                           train_file_indices,
                                           sp,ep,
                                           leehon_mapping,save_tag=args.save_tag,
                                           waveform_offset=10,
                                           return_waveforms=False,
                                           savedir=savedir)
        print "Finished get_processed_examples"
        estimate_templates((args.num_mix,),
                            Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.save_tag,
                            savedir=savedir)
    print "Finished estimate_templates"
    if args.save_detection_setup == "test":
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=save_detection_setup(num_mix,test_example_lengths,
                                     test_path,test_file_indices,args.detect_object,sp,
                                     ep,leehon_mapping,save_tag=args.save_tag,template_tag='train_2',savedir=args.savedir,verbose=args.v))
                jobs.append(p)
                p.start

                
    elif args.save_detection_setup == "train":
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=save_detection_setup(num_mix,train_example_lengths,
                             train_path,train_file_indices,args.detect_object,sp,
                             ep,leehon_mapping,save_tag=args.save_tag,template_tag='train_2',savedir=args.savedir,verbose=args.v))
                jobs.append(p)
                p.start
        else:
            save_detection_setup(args.num_mix,train_example_lengths,
                                 train_path,train_file_indices,args.detect_object,sp,
                                 ep,leehon_mapping,save_tag=args.save_tag,template_tag='train_2',savedir=args.savedir,verbose=args.v)

    if args.plot_detection_outs != '':
        print "Plotting the detection outputs"
        plot_detection_outs(args.plot_detection_outs,args.num_mix,sp,ep,test_path,test_file_indices,
                            save_tag=args.save_tag,template_tag='train_2',
                            savedir=args.savedir,
                            verbose=args.v)

    if args.get_fpr_tpr_tagged:
        print "Finished save_detection_setup"
    #get_estimated_detection_clusters(num_mix)
        get_fpr_tpr_tagged(args.num_mix,syllable_string,
                           return_detected_examples=False,
                           return_clusters=False,
                           save_tag=args.save_tag,
                           get_plots=True)
    if args.get_detection_clusters_for_2nd_stage:
        print "Getting detection clusters for 2nd stage"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=get_clustered_examples_for_2nd_stage(
                        num_mix,args.save_tag,args.savedir,
                        args.thresh_percent,
                        train_path,train_file_indices,syllable_string,
                        old_max_detect_tag=args.old_max_detect_tag))
                jobs.append(p)
                p.start
        else:
            get_clustered_examples_for_2nd_stage(args.num_mix,args.save_tag,args.savedir,
                                         args.thresh_percent,
                                         syllable_string,
                                             old_max_detect_tag=args.old_max_detect_tag)
    if args.train_second_stage_detectors:
        print "training second stage detectors"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=train_second_stage_detectors(num_mix,syllable_string,sp,ep,
                                 thresh_percent=args.thresh_percent,save_tag=args.save_tag,savedir=args.savedir,old_tag=args.old_max_detect_tag,
                                 make_plots=args.make_plots,
                                 waveform_offset=args.waveform_offset))
                jobs.append(p)
                p.start
        else:
            train_second_stage_detectors(num_mix,syllable_string,sp,ep,
                                 thresh_percent=args.thresh_percent,save_tag=args.save_tag,savedir=args.savedir,
                                 make_plots=args.make_plots,
                                 waveform_offset=args.waveform_offset)

def perform_test_phase_detection(num_mix,test_example_lengths,test_path,
                                 test_file_indices,
                                 syllable,sp,ep,leehon_mapping,
                                 save_tag,
                                 savedir,
                                 old_max_detect_tag,
                                 thresh_percent,
                                 verbose=False):
    save_detection_setup(num_mix,test_example_lengths,
                         test_path,test_file_indices,syllable,sp,
                         ep,leehon_mapping,save_tag="test")
    # going to use detection at the same threshold
    get_fpr_tpr_tagged(num_mix,syllable_string,
                return_detected_examples=False,
                return_clusters=False,
                save_tag='test',get_plots=True)
    test_detect_clusters= get_tagged_detection_clusters(num_mix,thresh_percent,save_tag='test',old_max_detect_tag='train')
    (pos_times,
     false_pos_times,
     false_neg_times,
     example_types,
     ) = get_detection_clusters_by_label(num_mix,test_path,
                                    test_file_indices,thresh_percent,single_threshold=True,save_tag='test',verbose=False,return_example_types=True,
                                         detect_clusters=test_detect_clusters)
    out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                      thresh_percent,'test'),'wb')
    pickle.dump(false_pos_times,out)
    out.close()
    out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                thresh_percent,'test'),'wb')
    pickle.dump(pos_times,out)
    out.close()
    out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                      thresh_percent,'test'),'wb')
    pickle.dump(false_neg_times,out)
    out.close()


    get_false_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='test',
                           savedir=savedir,
                           verbose=True)
    get_true_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='test',
                          savedir=savedir,
                           verbose=True)
    get_false_neg_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag='test',
                           savedir=savedir,
                           verbose=True)

    perform_second_stage_detection_testing(num_mix,syllable_string,'test',thresh_percent,
                                           make_plots=False,verbose=False)


def train_second_stage_detectors(num_mix,syllable_string,sp,ep,
                                 thresh_percent,save_tag,savedir,old_tag,
                                 make_plots=False,
                                 waveform_offset=10):
    get_false_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=waveform_offset,
                           thresh_percent=thresh_percent,save_tag=save_tag,
                           savedir=savedir,
                           verbose=False)
    run_fp_detector(num_mix,syllable_string,save_tag,make_plots=make_plots,
                    thresh_percent=thresh_percent,save_tag=old_tag,savedir=savedir,verbose=True)


def get_clustered_examples_for_2nd_stage(num_mix,save_tag,savedir,
                                         thresh_percent,train_path,train_file_indices,
                                         syllable_string,old_max_detect_tag='train'):
    get_fpr_tpr_tagged(num_mix,syllable_string,
                           return_detected_examples=False,
                           return_clusters=False,
                           save_tag=old_max_detect_tag,
                           get_plots=True)
    get_tagged_detection_clusters(num_mix,thresh_percent,save_tag=save_tag,
                                  old_max_detect_tag=old_max_detect_tag,
                                  savedir=savedir)
    (pos_times,
     false_pos_times,
     false_neg_times,
     example_types,
     ) = get_detection_clusters_by_label(num_mix,train_path,
                                    train_file_indices,thresh_percent,single_threshold=True,save_tag=save_tag,verbose=False,savedir=savedir,return_example_types=True)
    out = open('data/%s_false_pos_times_%d_%d_%s.pkl' % (syllable_string,num_mix,
                                                      thresh_percent,save_tag),'wb')
    pickle.dump(false_pos_times,out)
    out.close()
    out = open('data/%s_pos_times_%d_%d_%s.pkl' % (syllable_string,num_mix,
                                                thresh_percent,save_tag),'wb')
    pickle.dump(pos_times,out)
    out.close()
    out = open('data/%s_false_neg_times_%d_%d_%s.pkl' % (syllable_string,num_mix,
                                                      thresh_percent,save_tag),'wb')
    pickle.dump(false_neg_times,out)
    out.close()


def plot_detection_outs(plot_name,num_mix,sp,ep,data_path,file_indices,
                        save_tag='',template_tag='',savedir='data/',
                        verbose=False):
    # get the
    templates =get_templates(num_mix,template_tag=template_tag)
    C0 = int(np.mean([ len(k) for k in templates]))+1
    detection_array=np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    detection_template_ids= np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'rb')
    example_start_end_times=pickle.load(out)
    out.close()
    cur_plot=0
    for utt_id, utt in enumerate(example_start_end_times):

        utterance = gtrd.makeUtterance(data_path,file_indices[utt_id])
        S = gtrd.get_spectrogram(utterance.s,sp)
        E = gtrd.get_edge_features(S.T,ep,verbose=False
                                       )
        print utt_id,S.shape[0],utterance.flts[-1]
        for e_id,e in enumerate(utt):
            start_idx = max(e[0]-C0-2,0)
            end_idx=min(e[0]+C0+2,S.shape[0]-1)
            print e
            print zip(utterance.phns,utterance.flts)
            use_xticks,use_xticklabels = get_phone_xticks_locs_labels(start_idx,end_idx,utterance.flts,utterance.phns)
            use_xticks=np.append(use_xticks,e[0]-start_idx)
            use_xticklabels=np.append(use_xticklabels,'start')
            print e_id
            plt.close('all')
            plt.figure()
            plt.figtext(.5,.965,'Max Score is %f' % np.max(detection_array[utt_id,
                                                                 start_idx:end_idx]))
            ax1=plt.subplot(2,1,1)
            ax1.imshow(S[start_idx:end_idx].T,origin="lower left")
            ax1.set_xticks(use_xticks)
            ax1.set_xticklabels(use_xticklabels )
            ax2=plt.subplot(2,1,2,sharex=ax1)
            ax2.plot(np.arange(end_idx-start_idx),
                     detection_array[utt_id,start_idx:end_idx])
            ax2.set_xticks(use_xticks)
            ax2.set_xticklabels(use_xticklabels)
            plt.savefig('%s%s_%d_%d_%d.png' %(savedir,plot_name,
                                            cur_plot,
                                            utt_id,
                                            e_id))
            print "fname = %s%s_%d_%d_%d.png" %(savedir,plot_name,
                                            cur_plot,
                                            utt_id,
                                            e_id)
            cur_plot+=1
            plt.close('all')
            

def get_phone_xticks_locs_labels(start_idx,end_idx,flts,phns):
    middle_phns =np.arange(len(flts))[flts >= start_idx]
    if len(middle_phns) == 0:
        start_phn_idx =0
    else:
        start_phn_idx= middle_phns.min()
    if len(np.arange(len(flts))[flts >  end_idx])==0:
        end_phn_idx = len(flts)
    else:
        end_phn_idx =np.arange(len(flts))[flts >  end_idx].min()
    return flts[start_phn_idx:end_phn_idx]-start_idx, phns[start_phn_idx:end_phn_idx]
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='''
Program to run various experiments in detecting
syllables and tracking their performance
''')
    parser.add_argument('--detect_object',nargs='+',
                        default=('aa', 'r'),
                        type=str,
                        metavar="PHONE",
                        help="Phone or sequence of phones to use in an"
                        +" acoustic object detection experiment. To be entered with spaces between each phone.")
    parser.add_argument('--sample_rate',nargs=1,
                        type=int,metavar='N',
                        help="Sample rate for the acoustic signal being processed",
                        default=16000)
    parser.add_argument('--num_window_samples',
                        nargs=1,
                        default=320,
                        type=int,metavar='N',
                        help="Number of samples in the acoustic signal to use to compute one frame of the spectrogram")
    parser.add_argument('--num_window_step_samples',
                        nargs=1,
                        default=80,
                        type=int,metavar='N',
                        help="Number of acoustic samples to jump between successive frame computations in the spectrogram")
    parser.add_argument('--fft_length',nargs=1,default=512,
                        type=int,metavar='N',
                        help="Length of the vector we compute the discrete Fourier transform (DFT) over for the spectrogram, if this is greater than num_window_samples, we pad the vector with zeros")
    parser.add_argument('--kernel_length',nargs=1,default=7,
                        type=int,metavar='N',
                        help="Length of the kernel for performing smoothing on the spectrogram, this is similar to doing DCT smoothing on the spectrogram as is commonly practiced in speech recognition")
    parser.add_argument('--freq_cutoff',nargs=1,default=3000,
                        type=int,metavar='N',
                        help="Frequency cutoff for the computation to limit high frequency components and mimic the telephone situation")
    parser.add_argument('--use_mel',action='store_true',
                        help="Whether to use mel filters over the spectrogram for smoothing and making the frequency axis logarithmic above 1kHz")
    parser.add_argument('--block_length',nargs=1,default=40,
                        type=int,metavar='N',
                        help="Blocks that we compute the adaptive edge threshold over")
    parser.add_argument('--spread_length',nargs=1,default=1,
                        type=int,metavar='N',
                        help="Amount of spreading to do for the edge features")
    parser.add_argument('--edge_threshold_quantile',nargs=1,default=.7,
                        type=float,metavar='X',
                        help="Quantile to threshold the edges at, defaults to .7")
    parser.add_argument('--save_syllable_features_to_data_dir',
                        nargs=1,default='',
                        type=str,metavar='PATH',
                        help="If set to a path then this will attempt to save training data estimated using the parameters to the included path for later processing in an experiment.  This includes spectrograms, edgemaps, and waveforms. Defaults to ''")
    parser.add_argument('--make_plots',action='store_true',
                        help="Ubiquitous argument for whether or not to make plots in whatever functions are called. Defaults to False.")
    parser.add_argument('-v',action='store_true',
                        help="Ubiquitous command that says whether to make the program run in verbose mode with lots of printouts, defaults to False")
    parser.add_argument('-leehon_mapping',action='store_true',
                        help="Whether to use the mapping of the phones from Lee and Hon defaults to None")
    parser.add_argument('--num_mix',
                        type=int,metavar='N',default='2',
                        help="Number of mixture components to be used in the experiment defaults to 2")
    parser.add_argument('--savedir',
                        type=str,metavar='PATH',default='data/',
                        help="The directory which is used for saved features and as the temporary storage space for intermediate representations of the data defaults to data/")
    parser.add_argument('--save_tag',
                        type=str,metavar='str',default='train',
                        help="Tag for distinguishing different runs in the same directory from each other")
    parser.add_argument('--save_detection_setup',default='',
                        type=str,metavar='str',
                        help="Says whether to store the detection array and run save_detection_setup: values are either 'train' or 'test', this affects which set of the data is used")
    parser.add_argument('--estimate_templates',action='store_true',
                        help="Whether to run estimate_templates and save those templates or not")
    parser.add_argument('--plot_detection_outs',default="",type=str,
                        help="whether to run the plot_detection_outs runs with no arguments")
    parser.add_argument('--get_fpr_tpr_tagged',action="store_true",
                        help="whether get_fpr_tpr_tagged to run the plot_detection_outs runs with no arguments"
                        )
    parser.add_argument('--num_mix_parallel',default=[],nargs='*',
                        type=int,metavar='N',
                        help="possibly empty sequence of integers that say run this program for different mixture numbers concurrently")
    parser.add_argument('--get_detection_clusters_for_2nd_stage',
                        action="store_true",
                        help="Default is -1, if greater than zero will run functions for getting detection clusters for that percent")
    parser.add_argument('--old_max_detect_tag',default=None,type=str,
                        help="a string, this is None otherwise.  This string is the tag for when get_fpr_tpr_tagged was called and hence the tag attached to max_detect_vals when it was saved previously.")
    parser.add_argument('--thresh_percent',default=-1,type=int,metavar='N',
                        help="the threshold percentile to be used for determining the detection threshold")
    parser.add_argument('--log_file',default='main_multiprocessing.log',
                        type=str,metavar='Path',help="which logfile to use to look at the logging outputs")
    parser.add_argument('--waveform_offset',default=15,type=int,
                        metavar='N',help="Number of frames to pad the waveform vectors by in order to get samples of the sounds")
    parser.add_argument('--train_second_stage_detectors',action="store_true",
                        help="Include this flag if you want to call train_second_stage_detectors")
    syllable=('aa','r')
    threshval = 100
    make_plots =True
    print parser.parse_args()
    main(parser.parse_args())
