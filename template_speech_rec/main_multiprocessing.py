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
import template_speech_rec.get_mistakes as get_mistakes
import matplotlib.pyplot as plt
import parts, gmm_em
import pickle,collections,cPickle,os,itertools,re
import argparse
import multiprocessing
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from scipy.io import wavfile

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
               do_mfccs=False,
               no_use_dpss=False,
               mel_nbands=40,
               num_ceps=13,
               liftering=.6,
               include_energy=False,
               include_deltas=False,
               include_double_deltas=False,
               delta_window=9,
               do_freq_smoothing=True,
               block_length=40,
                            spread_length=1,
                            threshold=.7,
               magnitude_block_length=0,
               magnitude_and_edge_features=False,
               magnitude_features=False,
               use_parts=False,
               parts_path="/home/mark/Template-Speech-Recognition/"
                         + "Development/102012/"
                         + "E_templates.npy",
               parts_S_path=None,
               save_parts_S=False,
                bernsteinEdgeThreshold=12,
                spreadRadiusX=2,
                spreadRadiusY=2,
               root_path='/home/mark/Template-Speech-Recognition/',
               train_suffix='Data/Train/',
               test_suffix='Data/Test/',
               savedir='data/',
               mel_smoothing_kernel=-1,
               penalty_list=['unreg', '1.0',
                                                 'little_reg','0.1',
                                                 'reg', '0.05',
                                                 'reg_plus', '0.01',
                                                 'reg_plus_plus','0.001'],
               partGraph=None,
               spreadPartGraph=False):

    train_path = root_path+train_suffix
    test_path =root_path+test_suffix

    print "root_path = %s" % root_path
    print "train_path = %s" % train_path
    print "test_path = %s" % test_path
    print "savedir= %s"% savedir

    if os.path.exists('%strain_file_indices.npy' %savedir):
        train_file_indices = np.load('%strain_file_indices.npy' % savedir)
    else:
        train_file_indices = gtrd.get_data_files_indices(train_path)
        np.save('%strain_file_indices.npy' %savedir,train_file_indices)



    if os.path.exists('%stest_file_indices.npy' % savedir):
        test_file_indices = np.load('data/test_file_indices.npy')
    else:
        test_file_indices = gtrd.get_data_files_indices(test_path)
        np.save('%stest_file_indices.npy' %savedir,test_file_indices)


    if os.path.exists('%stest_example_lengths.npy' %savedir):
        test_example_lengths =np.load("%stest_example_lengths.npy" %savedir)
    else:
        test_example_lengths = gtrd.get_detect_lengths(test_file_indices,test_path)
        np.save("%stest_example_lengths.npy" % savedir,test_example_lengths)


    if os.path.exists('%stest_classify_lengths.npy' %savedir):
        test_classify_lengths = np.load("%stest_classify_lengths.npy" %savedir)
    else:
        test_classify_lengths = gtrd.get_classify_lengths(test_file_indices,
                                                          test_path)
        np.save("%stest_classify_lengths.npy" %savedir,test_classify_lengths)

    if not os.path.exists('%stest_classify_labels.npy' %savedir):
        test_classify_labels = gtrd.get_classify_labels(test_file_indices,
                                                          test_path,test_classify_lengths)
        np.save("%stest_classify_labels.npy" %savedir,test_classify_labels)



    if os.path.exists('%strain_example_lengths.npy' %savedir):
        train_example_lengths =np.load("%strain_example_lengths.npy" %savedir)
    else:
        train_example_lengths = gtrd.get_detect_lengths(train_file_indices,train_path)
        np.save("%strain_example_lengths.npy" %savedir,train_example_lengths)

    if os.path.exists('%strain_classify_lengths.npy' %savedir):
        print "loaded train classify lengths"
        train_classify_lengths = np.load("%strain_classify_lengths.npy" %savedir)
    else:
        print "getting train classify lengths"
        train_classify_lengths = gtrd.get_classify_lengths(train_file_indices,
                                                          train_path)
        np.save("%strain_classify_lengths.npy" %savedir,train_classify_lengths)


    if not os.path.exists('%strain_classify_labels.npy' %savedir):
        train_classify_labels = gtrd.get_classify_labels(train_file_indices,
                                                          train_path,train_classify_lengths)
        np.save("%strain_classify_labels.npy" %savedir,train_classify_labels)


    if use_parts:
        # get the parts Parameters
        EParts=np.clip(np.load(parts_path),.01,.99)
        logParts=np.log(EParts).astype(np.float64)
        logInvParts=np.log(1-EParts).astype(np.float64)
        numParts =EParts.shape[0]
        if partGraph is not None and spreadPartGraph:
            partGraph = np.load(partGraph).astype(np.uint8)
        else:
            partGraph = None
        if save_parts_S and parts_S_path is not None:
            parts_S = np.load(parts_S_path)
        else:
            parts_S_path= None
            parts_S = None
        pp =gtrd.makePartsParameters(
            use_parts=use_parts,
            parts_path=parts_path,
            bernsteinEdgeThreshold=bernsteinEdgeThreshold,
                                   logParts=logParts,
                                   logInvParts=logInvParts,
                                   spreadRadiusX=spreadRadiusX,
                                   spreadRadiusY=spreadRadiusY,
                                   numParts=numParts,
            partGraph=partGraph,
            parts_S=parts_S)
    else:
        pp=None


    if magnitude_block_length ==0:
        magnitude_block_length = block_length

    return (gtrd.makeSpectrogramParameters(
            sample_rate=sample_rate,
            num_window_samples=num_window_samples,
            num_window_step_samples=num_window_step_samples,
            fft_length=fft_length,
            kernel_length=kernel_length,
            freq_cutoff=freq_cutoff,
            use_mel=use_mel,
            mel_smoothing_kernel=mel_smoothing_kernel,
            do_mfccs=do_mfccs,
            nbands=mel_nbands,
            num_ceps=num_ceps,
            liftering=liftering,
            include_energy=include_energy,
            include_deltas=include_deltas,
            include_double_deltas=include_double_deltas,
            delta_window=delta_window,
            no_use_dpss=no_use_dpss,
            do_freq_smoothing=do_freq_smoothing,
            ),
            gtrd.makeEdgemapParameters(block_length=block_length,
                                        spread_length=spread_length,
                                        threshold=threshold,
                                       magnitude_features=magnitude_features,
                                       magnitude_block_length=magnitude_block_length,
                                       magnitude_and_edge_features=magnitude_and_edge_features),
            pp,
            root_path,
            test_path,train_path,
            train_example_lengths, train_file_indices,train_classify_lengths,
            test_example_lengths, test_file_indices,test_classify_lengths,

            zip(penalty_list[::2],(float(k) for k in  penalty_list[1::2])))


#
# need to access the files where we perform the estimation
#

def get_leehon39_dict(no_sil=False):
    """
    Output:
    ======
    leehon_mapping:
       dictionary with the 39 phone classes from leehon paper
    rejected_phones:
       rejected phones
    use_phns:
       use_phns
    """
    leehon_groups = {
        'sil': ['h#','#h','pau','bcl','dcl','gcl','pcl','tcl','kcl','qcl','epi'],
        'iy': ['iy'],
        'ih': ['ih','ix'],
        'eh': ['eh'],
        'ae': ['ae'],
        'ax': ['ax','ah','ax-h'],
        'uw': ['uw','ux'],
        'uh': ['uh'],
        'ao': ['ao','aa'],
        'ey': ['ey'],
        'ay': ['ay'],
        'oy': ['oy'],
        'aw': ['aw'],
        'ow': ['ow'],
        'l': ['l','el'],
        'r': ['r'],
        'y': ['y'],
        'w': ['w'],
        'er': ['axr','er'],
        'm': ['m','em'],
        'n': ['n','nx','en'],
        'ng': ['ng','eng'],
        'ch': ['ch'],
        'jh': ['jh'],
        'dh': ['dh'],
        'b': ['b'],
        'd':['d'],
        'dx': ['dx'],
        'g': ['g'],
        'p': ['p'],
        't': ['t'],
        'k': ['k'],
        'z': ['z'],
        'zh': ['zh','sh'],
        'v': ['v'],
        'f': ['f'],
        'th': ['th'],
        's': ['s'],
        'hh': ['hh','hv']
        }

    rejected_phns = ['q']
    if no_sil:
        rejected_phns.append('sil')
        rejected_phns.extend(leehon_groups['sil'])
        a = leehon_groups.pop('sil',None)
    leehon_mapping = dict(reduce(lambda x,y: x+y,
                                 (tuple( (v_elem,k)
                                        for v_elem in v)
                                 for k,v in leehon_groups.items())))
    return leehon_mapping, rejected_phns,leehon_groups.keys()

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


def save_all_leehon_phones(utterances_path,file_indices,leehon_mapping,phn,
                           sp,ep,pp,save_tag,savedir,mel_smoothing_kernel,
                           offset,waveform_offset,num_use_file_idx=-1):
    """
    Parameters:
    ==========
    Output:
    =======
    """


    if num_use_file_idx == -1:
        num_use_file_idx = len(file_indices)

    phn_features,avg_bgd, avg_spec_bgd=gtrd.get_syllable_features_directory(
        utterances_path,
        file_indices[:num_use_file_idx],
        (phn,),
        S_config=sp,E_config=ep,offset=offset,
        E_verbose=False,return_avg_bgd=True,
        waveform_offset=15,
        phn_mapping=leehon_mapping,
        P_config=pp,
        mel_smoothing_kernel=mel_smoothing_kernel,
        do_avg_bgd_spec=True)
    bgd = np.clip(avg_bgd.E,.01,.4)
    avg_spec_bgd = avg_spec_bgd.E
    np.save('%sbgd_%s.npy' % (savedir,save_tag),bgd)
    np.save('%sspec_bgd_%s.npy' % (savedir,save_tag),avg_spec_bgd)

    example_mat = gtrd.recover_example_map(phn_features)

    avg_bgd_std = gtrd.AverageBackground()
    for e in phn_features:
        if len(e) > 0: break

    for fl in file_indices[:num_use_file_idx]:
        utterance = gtrd.makeUtterance(e[0].utt_path,e[0].file_idx,
                                           use_noise_file=None,
                                           noise_db=None)

        S = gtrd.get_spectrogram(utterance.s,sp)
        S -= avg_spec_bgd
        avg_bgd_std.add_frames(S**2,time_axis=0)

    avg_bgd_sigma = avg_bgd_std.E * ( avg_bgd.num_frames/(avg_bgd.num_frames+1.))
    np.save('%sspec_bgd_sigma_%s.npy' % (savedir,save_tag),avg_bgd_sigma)


    Slengths,Ss  = gtrd.recover_specs(phn_features,example_mat,bgd=avg_spec_bgd,bgd_std=avg_bgd_sigma)
    Ss = Ss.astype(np.float32)

    np.savez('%sSs_lengths_%s_%s.npz' % (savedir,
                                             phn,
                                             save_tag),Ss=Ss,Slengths=Slengths,example_mat=example_mat)

    Elengths,Es  = gtrd.recover_edgemaps(phn_features,example_mat,bgd=bgd)
    Es = Es.astype(np.uint8)
    np.savez('%sEs_lengths_%s_%s.npz'% (savedir,
                                        phn,
                                        save_tag) ,Es=Es,Elengths=Elengths,example_mat=example_mat)




def save_syllable_features_to_data_dir(phn_tuple,
                          utterances_path,
                          file_indices,

                         sp,ep,
                          phn_mapping,pp=None,tag_data_with_syllable_string=False,
                                       save_tag="train",
                          waveform_offset=10,
                                       block_features=False,
                                       savedir='data/',verbose=False,
                                       mel_smoothing_kernel=-1,
                                       offset=0,
                                       num_use_file_idx=-1):
    """
    Wrapper function to get all the examples processed
    """
    print "Collecting the data for phn_tuple " + ' '.join('%s' % k for k in phn_tuple)
    syllable_string = '_'.join(p for p in phn_tuple)

    if verbose:
        print "will save waveforms to %s" % ('%s%s_waveforms_lengths_%s.npz' % (savedir,
                                                    syllable_string,
                                                       save_tag))
        print "will save Ss to %s" % ('%sSs_lengths_%s.npz' % (
                savedir,
                                             save_tag))
        print "will save Es to %s" % ('%sEs_lengths_%s.npz' % (
                savedir,
                                             save_tag))


    if num_use_file_idx == -1:
        num_use_file_idx = len(file_indices)

    phn_features,avg_bgd, avg_spec_bgd=gtrd.get_syllable_features_directory(
        utterances_path,
        file_indices[:num_use_file_idx],
        phn_tuple,
        S_config=sp,E_config=ep,offset=offset,
        E_verbose=False,return_avg_bgd=True,
        waveform_offset=15,
        phn_mapping=phn_mapping,
        P_config=pp,
        verbose=verbose,
        mel_smoothing_kernel=mel_smoothing_kernel,
        do_avg_bgd_spec=True)
    bgd = np.clip(avg_bgd.E,.01,.4)
    avg_spec_bgd = avg_spec_bgd.E
    np.save('%sbgd_%s.npy' % (savedir,save_tag),bgd)
    np.save('%sspec_bgd_%s.npy' % (savedir,save_tag),avg_spec_bgd)

    if verbose:
        print  "Estimating the standard deviation"
    avg_bgd_std = gtrd.AverageBackground()
    for e in phn_features:
        if len(e) > 0: break

    for fl in file_indices[:num_use_file_idx]:
        utterance = gtrd.makeUtterance(e[0].utt_path,e[0].file_idx,
                                           use_noise_file=None,
                                           noise_db=None)

        S = gtrd.get_spectrogram(utterance.s,sp)
        S -= avg_spec_bgd
        avg_bgd_std.add_frames(S**2,time_axis=0)

    avg_bgd_sigma = avg_bgd_std.E * ( avg_bgd.num_frames/(avg_bgd.num_frames+1.))
    np.save('%sspec_bgd_sigma_%s.npy' % (savedir,save_tag),avg_bgd_sigma)

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


    Slengths,Ss  = gtrd.recover_specs(phn_features,example_mat,bgd=avg_spec_bgd,bgd_std=avg_bgd_sigma)
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
                           savedir='data/',):
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

def visualize_bern_on_specs(fl_path,E,S,part_id,E_id=None,part_S=None):
    S_view = S[:E.shape[0],
                :E.shape[1]]
    plt.close('all')
    plt.figure()
    plt.clf()

    if part_S is None:
        plt.imshow(S_view.T.astype(np.float),aspect=.3,origin="lower",alpha=.8)
        plt.imshow(E[:,:,part_id].T.astype(np.float),cmap=cm.bone,vmin=0,vmax=1,origin="lower left",alpha = .4,aspect=.3)
        if E_id is not None:
            plt.title('Fl:%d, part:%d' %(E_id,part_id))
        else:
            plt.title('part:%d' %part_id)
        plt.axis('off')
    else:
        plt.subplot(2,1,1)
        plt.imshow(S_view.T.astype(np.float),aspect=.3,origin="lower",alpha=.8)
        plt.imshow(E[:,:,part_id+1].T.astype(np.float),cmap=cm.bone,vmin=0,vmax=1,origin="lower left",alpha = .4,aspect=.3)
        if E_id is not None:
            plt.title('Fl:%d, part:%d' %(E_id,part_id))
        else:
            plt.title('part:%d' %part_id)
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.imshow(part_S,cmap=cm.bone,interpolation='nearest')
        plt.axis('off')
    plt.savefig(fl_path)
    plt.close('all')


def visualize_processed_examples(Es,Elengths,Ss,Slengths,syllable_string='aa_r',
                                 savedir='data/',plot_prefix='vis_bern',pp=None):
    for E_id, E in enumerate(Es):
        for part_id in xrange(E.shape[-1]):
            if pp is None:
                visualize_bern_on_specs('%s%s_%s_%d_%d.png' % (savedir,
                                                               syllable_string,
                                                               plot_prefix,
                                                               E_id,part_id),
                                        E[:Elengths[E_id]],Ss[E_id][:Slengths[E_id]],part_id)
            elif pp.parts_S is None:
                visualize_bern_on_specs('%s%s_%s_%d_%d.png' % (savedir,
                                                               syllable_string,
                                                               plot_prefix,
                                                               E_id,part_id),
                                        E[:Elengths[E_id]],Ss[E_id][:Slengths[E_id]],part_id)
            else:
                visualize_bern_on_specs('%s%s_%s_%d_%d.png' % (savedir,
                                                               syllable_string,
                                                               plot_prefix,
                                                               E_id,part_id),
                                        E[:Elengths[E_id]],Ss[E_id][:Slengths[E_id]],part_id,
                                        part_S=pp.parts_S[part_id])


def visualize_template(num_mix_parallel,syllable_string='aa_r',
                       template_tag='',
                                 savedir='data/',plot_prefix='vis_template'):
    max_length=0
    for num_mix in num_mix_parallel:
        if num_mix > 1:
            outfile = np.load('%s%d_spec_templates_%s.npz' % (savedir,num_mix,template_tag))
            templates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
        else:
            templates = (np.load('%s1_spec_templates_%s.npy' % (savedir,template_tag))[0] ,)
        max_length=max(max_length,
                       max(len(t) for t in templates))


    for num_mix in num_mix_parallel:
        if num_mix > 1:
            outfile = np.load('%s%d_spec_templates_%s.npz' % (savedir,num_mix,template_tag))
            Stemplates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
            outfile = np.load('%s%d_templates_%s.npz' % (savedir,num_mix,template_tag))
            Etemplates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))

        else:
            Stemplates = (np.load('%s1_spec_templates_%s.npy' % (savedir,template_tag))[0] ,)
            Etemplates = (np.load('%s1_templates_%s.npy' % (savedir,template_tag))[0] ,)
        for S_idx,SE in enumerate(itertools.izip(Stemplates,Etemplates)):
            S,E=SE
            if len(S) == max_length:
                view_S = S
                view_E = E
            else:
                view_S = np.vstack((S,
                                    S.min() * np.ones(
                            (max_length-len(S),) + S.shape[1:])))
                view_E = np.vstack((E,
                                    E.min() * np.ones(
                            (max_length-len(E),) + E.shape[1:],
                            )))

            view_E=view_E.swapaxes(2,1).swapaxes(0,1)


            plt.close('all')

            for plt_id in xrange(view_E.shape[0]/8):
                plt.figure()

                for i in xrange(view_E.shape[0]):
                    plt.subplot(4,2,i+1)
                    plt.imshow(view_S.T.astype(np.float),aspect=.3,
                               cmap=cm.bone,
                               origin="lower",
                               alpha=.8)
                    plt.imshow(view_E[plt_id* view_E.shape[0]+i].T,vmin=0,vmax=1,
                               origin="lower left",alpha=.4,aspect=.3)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig('%s%s_%s_%s_%d_%d_%d_pos.png' % (savedir,
                                              syllable_string, template_tag,
                                              plot_prefix,
                                              num_mix,S_idx,plt_id))
                plt.close('all')
                plt.figure()
                for i in xrange(view_E.shape[0]):
                    plt.subplot(4,2,i+1)
                    plt.imshow(view_S.T.astype(np.float),aspect=.3,cmap=cm.bone,
                               origin="lower",
                               alpha=.8)
                    plt.imshow(1-view_E[plt_id* view_E.shape[0]+i].T,vmin=0,vmax=1,
                           origin="lower left",alpha=.4,aspect=.3)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig('%s%s_%s_%s_%d_%d_%d_neg.png' % (savedir,
                                                          syllable_string, template_tag,
                                                          plot_prefix,
                                                          num_mix,S_idx,plt_id))
                plt.close('all')




def estimate_templates(num_mix_params,
                       Es,Elengths,
                       Ss,Slengths,
                       get_plots=False,save_tag='',
                       savedir='data/',do_truncation=True,
                       percent_use=None,
                       percent_use_seed=0,
                       template_tag=None):
    if template_tag is None:
        template_tag = save_tag
    f = open('%smixture_estimation_stats_regular.data' % savedir ,'w')
    if percent_use is not None and percent_use < 1:
        np.random.seed(percent_use_seed)
        use_idx = np.random.permutation(len(Es))[: int(percent_use * len(Es))]
        print "Using %f fraction of the data, totals to %d observations" % (percent_use, len(use_idx))
        Es = Es[use_idx]
        Elengths = Elengths[use_idx]
        Ss = Ss[use_idx]
        Slengths = Slengths[use_idx]


    for num_mix in num_mix_params:
        print num_mix
        if num_mix == 1:
            affinities = np.ones((Es.shape[0],1),dtype=np.float64)
            mean_length = int(np.mean(Elengths) + .5)
            templates = (np.mean(Es,0)[:mean_length],)
            spec_templates = (np.mean(Ss,0)[:mean_length],)
            np.save('%s%d_affinities_%s.npy' % (savedir,num_mix,template_tag),
                    affinities)
            np.save('%s%d_templates_%s.npy' % (savedir,num_mix,template_tag),
                    templates)
            np.save('%s%d_spec_templates_%s.npy' % (savedir,num_mix,template_tag),
                    spec_templates)
            np.save('%s%d_templates_%s.npy' % (savedir,num_mix,template_tag),
                    templates)
            np.save('%s%d_spec_templates_%s.npy' % (savedir,num_mix,template_tag),
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
                                                                  Elengths,
                                                              do_truncation=do_truncation)
            spec_templates = et.recover_different_length_templates(bem.affinities,
                                                                       Ss,
                                                                       Slengths,
                                                                   do_truncation=do_truncation)
            if get_plots:
                plt.close('all')
                for i in xrange(num_mix):
                    plt.figure()
                    plt.imshow(spec_templates[i].T,origin="lower left")
                    plt.savefig('%s%d_spec_templates_%d_%s.png' % (savedir,num_mix,i,template_tag))
                    plt.close()

            np.save('%s%d_affinities_%s.npy' % (savedir,num_mix,template_tag),
                    bem.affinities)
            np.savez('%s%d_templates_%s.npz' % (savedir,num_mix,template_tag),
                     *templates)
            np.savez('%s%d_spec_templates_%s.npz' % (savedir,num_mix,template_tag),
                     *spec_templates)
            np.savez('%s%d_templates_%s.npz' % (savedir,num_mix,template_tag),
                     *templates)
            np.savez('%s%d_spec_templates_%s.npz' % (savedir,num_mix,template_tag),
                     *spec_templates)
            f.write('%d %d ' % (num_mix,
                                len(bem.affinities))
                + ' '.join(str(np.sum(np.argmax(bem.affinities,1)==i))
                           for i in xrange(bem.affinities.shape[1]))
                    +'\n')
    f.close()


def estimate_spectral_templates(num_mix_params,
                                Es,Elengths,
                                Ss,Slengths,
                                get_plots=False,save_tag='',
                                savedir='data/',do_truncation=True):
    f = open('%smixture_estimation_stats_regular.data' % savedir ,'w')

    for num_mix in num_mix_params:
        print num_mix
        if num_mix == 1:
            affinities = np.ones((Ss.shape[0],1),dtype=np.float64)
            mean_length = int(np.mean(Slengths) + .5)
            templates = (np.mean(Ss,0)[:mean_length],)
            spec_sigmas = (np.mean((Ss - np.mean(Ss,0))**2,0)[:mean_length],)
            E_templates = et.recover_different_length_templates(affinities,
                                                                  Es,
                                                                  Elengths,
                                                              do_truncation=do_truncation)

            np.save('%s%d_affinities_%s.npy' % (savedir,num_mix,save_tag),
                    affinities)
            np.savez('%s%d_spec_templates_%s.npz' % (savedir,num_mix,save_tag),
                    *templates)
            np.savez('%s%d_spec_sigmas_%s.npz' % (savedir,num_mix,save_tag),
                    *spec_sigmas)
            np.savez('%s%d_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *E_templates)
            #
            # write the data to the mixture file for checking purposes
            # format is:
            #   num_components total c0 c1 c2 ... ck
            f.write('%d %d %g\n' % (num_mix,
                                    len(affinities),np.sum(affinities[:,0])))
        else:
            centers = gmm_em.kmeanspp_gmm_init(num_mix,Ss)
            centers,sigmas,weights,membership_probs = gmm_em.GMM_EM(Ss,centers,tol=.000001)




            templates = et.recover_different_length_templates(membership_probs,
                                                                  Es,
                                                                  Elengths,
                                                              do_truncation=do_truncation)

            spec_templates, out_sigmas = et.recover_different_length_templates(membership_probs,
                                                                       Ss,
                                                                       Slengths,
                                                                   do_truncation=do_truncation,sigmas=sigmas)
            if get_plots:
                plt.close('all')
                for i in xrange(num_mix):
                    plt.figure()
                    plt.imshow(spec_templates[i].T,origin="lower left")
                    plt.savefig('%s%d_spec_templates_%d_%s.png' % (savedir,num_mix,i,save_tag))
                    plt.close()

            np.save('%s%d_affinities_%s.npy' % (savedir,num_mix,save_tag),
                    membership_probs)
            np.savez('%s%d_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *templates)
            np.savez('%s%d_spec_templates_%s.npz' % (savedir,num_mix,save_tag),
                     *spec_templates)
            np.save('%s%s_spec_weights_%s.npy' % (savedir,num_mix,save_tag),
                    weights)
            np.savez('%s%d_spec_sigmas_%s.npz' % (savedir,num_mix,save_tag),
                     *out_sigmas)

            f.write('%d %d ' % (num_mix,
                                len(membership_probs))
                + ' '.join(str(np.sum(np.argmax(membership_probs,1)==i))
                           for i in xrange(membership_probs.shape[1]))
                    +'\n')
    f.close()



def get_templates(num_mix,template_tag=None,savedir='data/',
                  clip_factor=.001,use_svm_based=False,svm_name=None,syllable_string=None,use_spectral=False):

    if template_tag is None:
        if num_mix > 1:
            outfile = np.load('%s%d_templates_regular.npz' % (savedir,num_mix))
            print "template loading %s%d_templates_regular.npz" % (savedir,num_mix)
            templates = tuple( np.clip(outfile['arr_%d'%i],clip_factor,
                                       1-clip_factor) for i in xrange(len(outfile.files)))
        else:
            templates = (np.clip(np.load('%s1_templates_regular.npy' % savedir)[0],
                         clip_factor,1-clip_factor),)
            print 'template loading %s1_templates_regular.npy' % savedir
    elif use_svm_based and svm_name is not None and syllable_string is not None:
        print "using svm based template"
        if num_mix > 1:
            outfile = np.load('%s%s_svm_based_lrt_templates_%d_%s_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               svm_name,template_tag))
            print 'template loading: %s%s_svm_based_lrt_templates_%d_%s_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               svm_name,template_tag)
            templates = tuple(np.clip(outfile['arr_%d' %i],clip_factor,
                                      1-clip_factor) for i in xrange(len(outfile.files)))
        else:
            templates = (np.clip(np.load('%s%s_svm_based_lrt_templates_%d_%s_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               svm_name,template_tag)),clip_factor,
                                      1-clip_factor),)
            print 'template loading %s%s_svm_based_lrt_templates_%d_%s_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               svm_name,template_tag)
    elif use_spectral:
        print "using spectrum based template"
        outfile = np.load('%s%d_spec_templates_%s.npz' % (savedir,

                                                          num_mix,
                                                               template_tag))
        print 'template loading: %s%d_spec_templates_%s.npz' % (savedir,

                                                          num_mix,
                                                               template_tag)
        templates = tuple(outfile['arr_%d' %i]
                          for i in xrange(len(outfile.files)))

        outfile=np.load('%s%d_spec_sigmas_%s.npz' % (savedir,

                                                          num_mix,
                                                               template_tag))
        sigmas = tuple(outfile['arr_%d' %i]
                          for i in xrange(len(outfile.files)))

    else:
        if num_mix > 1:
            outfile = np.load('%s%d_templates_%s.npz' % (savedir,num_mix,template_tag))
            print 'template loading: %s%d_templates_%s.npz' % (savedir,num_mix,template_tag)
            templates = tuple( np.clip(outfile['arr_%d'%i],
                                       clip_factor,1-clip_factor) for i in xrange(len(outfile.files)))

        else:
            templates = (np.clip(np.load('%s1_templates_%s.npy' % (savedir,template_tag))[0] ,
                                 clip_factor,1-clip_factor),)

            print 'template loading %s1_templates_%s.npy' % (savedir,template_tag)
    if use_spectral:
        return templates, sigmas
    else:
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

def visualize_detection_setup(num_mix,data_example_lengths,
                         data_path,file_indices,sp,prefix_tag,
                         savedir,save_tag):

    detection_array= np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    detection_template_ids= np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    detection_lengths=np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'rb')
    example_start_end_times=pickle.load(out)
    out.close()


    for fl_id, fl in enumerate(file_indices):
        utterance = gtrd.makeUtterance(data_path,fl)
        S = gtrd.get_spectrogram(utterance.s,sp)
        detect_scores = np.hstack((detection_array[fl_id][:detection_lengths[fl_id]],
                                   np.zeros(len(S) -detection_lengths[fl_id])))
        #import pdb; pdb.set_trace()
        use_xticks,use_xticklabels = (utterance.flts[:-1],utterance.phns)
        for e_id, e in enumerate(example_start_end_times[fl_id]):
            use_xticks=np.append(use_xticks,e[0])
            use_xticklabels=np.append(use_xticklabels,'start')
            use_xticks=np.append(use_xticks,e[1])
            use_xtickslabels=np.append(use_xticklabels,'end')

        plt.close('all')
        plt.figure()
        plt.figtext(.7,.465,'Max Score is %f,num_mix=%d,fl_id=%d' % (np.max(detection_array[fl_id,
                                                                 :detection_lengths[fl_id]]),num_mix,fl_id))
        ax1=plt.subplot(2,1,1)
        ax1.imshow(S.T,origin="lower left")
        ax1.set_xticks(use_xticks)
        ax1.set_xticklabels(use_xticklabels )
        ax2=plt.subplot(2,1,2,sharex=ax1)
        ax2.plot(np.arange(len(detect_scores)),
                 detect_scores)
        ax2.set_xticks(use_xticks)
        ax2.set_xticklabels(use_xticklabels)
        plt.savefig('%s%s_%d_%d.png' % (savedir,prefix_tag,
                                           fl_id,
                                        num_mix
                                           ))
        print "fname = %s%s_%d_%d.png" % (savedir,prefix_tag,
                                           fl_id,
                                          num_mix
                                           )
        plt.close('all')

        for e_id,e in enumerate(example_start_end_times[fl_id]):
            start_idx = max(e[0]-20,0)
            end_idx = min(e[0]+60,utterance.flts[-1])
            use_xticks,use_xticklabels = get_phone_xticks_locs_labels(start_idx,end_idx,utterance.flts,utterance.phns)
            use_xticks=np.append(use_xticks,e[0]-start_idx)
            use_xticklabels=np.append(use_xticklabels,'start')
            plt.close('all')
            plt.figure()
            plt.figtext(.5,.965,'Max Score is %f,num_mix=%d,fl_id=%d' % (np.max(detection_array[fl_id,
                                                                 start_idx+10:end_idx-20]),num_mix,fl_id))
            ax1=plt.subplot(2,1,1)
            ax1.imshow(S[start_idx:end_idx].T,origin="lower left")
            ax1.set_xticks(use_xticks)
            ax1.set_xticklabels(use_xticklabels )
            ax2=plt.subplot(2,1,2,sharex=ax1)
            ax2.plot(
                     detection_array[fl_id,start_idx:end_idx])
            ax2.set_xticks(use_xticks)
            ax2.set_xticklabels(use_xticklabels)
            plt.savefig('%sindividual_%s_%d_%d_%d.png' %(savedir,prefix_tag,
                                            fl_id,
                                            e_id,
                                                          num_mix))
            print "fname = %sindividual_%s_%d_%d_%d.png" %(savedir,prefix_tag,
                                            fl_id,
                                            e_id,
                                                            num_mix)


def get_first_layer_cascade(num_mix,syllable_string,template_tag,
                            savedir):
    templates =get_templates(num_mix,template_tag=template_tag,savedir=savedir)
    bgd = np.load('%sbgd_%s.npy' %(savedir,template_tag))
    linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd)
    np.savez('%s%s_first_layer_cascade_%d_%s.npz' % (savedir,
                                                        syllable_string,
                                                        num_mix,
                                                        template_tag),
             *(reduce(lambda x,y: x+y,linear_filters_cs)))



def run_detection_cascade_over_files(num_mix,test_example_lengths,
                          test_path,file_indices,syllable_string,sp,
                          ep,leehon_mapping,thresh_percent,
                          pp=None,
                          save_tag='',template_tag=None,
                          savedir='data/',
                          verbose=False,
                          num_use_file_idx=-1):
    bgd=np.load('%sbgd_$s.npy' % (savedir,template_tag))
    # these are the first layers of the cascade
    templates =get_templates(num_mix,template_tag=template_tag,savedir=savedir)
    detection_array = np.zeros((test_example_lengths.shape[0],
                            test_example_lengths.max() + 2),dtype=np.float32)

    outfile = np.load('%s%s_first_layer_cascade_%d_%d_%s.npz' % (savedir,
                                                          syllable_string,
                                                          num_mix,
                                                          thresh_percent,
                                                          template_tag))

    for fl_idx,fl in enumerate(file_indices):
        utterance = makeUtterance(test_path,fl_idx)
        sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
        S = get_spectrogram(utterance.s,sp)
        S_flts = utterance.flts
        E = get_edge_features(S.T,ep,verbose=False)
        E_flts = S_flts
        if pp is not None:
            # E is now the part features
            E = get_part_features(E,pp,verbose=False)
            E_flts = S_flts.copy()
            E_flts[-1] = len(E)


def save_second_layer_cascade(num_mix,syllable_string,
                              save_tag,template_tag,savedir,
                              num_binss=np.array([0,3,4,5,7,10,15,23]),
                              penalty_list=(('unreg', 1),
                                                 ('little_reg',.1),
                                                 ('reg', 0.05),
                                                 ('reg_plus', 0.01),
                                                 ('reg_plus_plus',.001)),

                              verbose=False,
                              only_svm=False,
                              use_spectral=False,
                              old_max_detect_tag=None,
                              load_data_tag=None):
    if old_max_detect_tag is None:
        old_max_detect_tag = template_tag
    cascade_layer = ()
    cascade_names = ()

    outfile = np.load('%slinear_filter_%d_%s.npz' % ( savedir,
                                                      num_mix,
                                                      template_tag))
    baseline_linear_filters = tuple(outfile['arr_%d' %i] for i in xrange(len(outfile.files)))
    outfile = np.load('%sc_%d_%s.npz'%(savedir,num_mix,template_tag))
    baseline_linear_filter_biases = tuple(outfile['arr_%d' %i] for i in xrange(len(outfile.files)))

    for mix_component in xrange(num_mix):

        cascade_layer += ((mix_component,
                           baseline_linear_filters[mix_component],
                           baseline_linear_filter_biases[mix_component],
                           '%s_baseline_%d_%s' % (syllable_string,

                                                  num_mix,
                                                  template_tag)),)

        cascade_names += ('%s_baseline_%d_%d_%s' % (syllable_string,
                                                     mix_component,
                                                      num_mix,
                                                             template_tag),)

        if not only_svm:
            for num_bins in num_binss:
                outfile = np.load('%s%s_lf_c_quantized_second_stage_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,num_bins,
                                                             template_tag))
                if verbose:
                    print '%s%s_lf_c_quantized_second_stage_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,num_bins,
                                                             template_tag)
                cascade_layer += ((mix_component,outfile['lf'],outfile['c'],
                               '%s_lf_c_quantized_second_stage_%d_%d_%s' % (syllable_string,

                                                                                     num_mix,num_bins,
                                                             template_tag)),)
                if verbose:
                    print '%s_lf_c_quantized_second_stage_%d_%d_%s' % (syllable_string,

                                                                                     num_mix,num_bins,
                                                             template_tag)
                cascade_names += ('%s_lf_c_quantized_second_stage_%d_%d_%d_%s' % (syllable_string,
                                                     mix_component,
                                                      num_mix,num_bins,
                                                             template_tag),)

                if verbose:
                    print '%s_lf_c_quantized_second_stage_%d_%d_%d_%s' % (syllable_string,
                                                     mix_component,
                                                      num_mix,num_bins,
                                                             template_tag)

            outfile = np.load('%s%s_lf_c_second_stage_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                                   num_mix,template_tag))
            if verbose:
                print '%s%s_lf_c_second_stage_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                                   num_mix,template_tag)
            cascade_layer += ((mix_component,outfile['lf'],outfile['c'],
                           '%s_lf_c_second_stage_%d_%s' % (syllable_string,
                                                                   num_mix,template_tag)),)
            if verbose:
                print '%s_lf_c_second_stage_%d_%s' % (syllable_string,
                                                                   num_mix,template_tag)
            cascade_shape = cascade_layer[-1][1].shape
            cascade_names += ('%s_lf_c_second_stage_%d_%d_%s' % (syllable_string,
                                                     mix_component,
                                                      num_mix,template_tag),)
            if verbose:
                print '%s_lf_c_second_stage_%d_%d_%s' % (syllable_string,
                                                     mix_component,
                                                      num_mix,template_tag)

        template_out = get_templates(num_mix,template_tag=template_tag,savedir=savedir, use_spectral=use_spectral)

        if use_spectral:

            templates,sigmas = template_out
        else:
            templates=template_out

        cascade_shape = templates[mix_component].shape
        for name, penalty in penalty_list:
            try:
                outfile = np.load('%s%s_w_b_second_stage_%d_%d_%s_%s.npz' % (savedir,syllable_string,
                                                          mix_component,
                                                          num_mix,
                                                               name,template_tag))
                if verbose:
                    print '%s%s_w_b_second_stage_%d_%d_%s_%s.npz' % (savedir,syllable_string,
                                                          mix_component,
                                                          num_mix,
                                                               name,template_tag)
            except:
                import pdb; pdb.set_trace()
            try:
                cascade_layer += ((mix_component,outfile['w'].reshape(cascade_shape),
                                   outfile['b'],
                               '%s_w_b_second_stage_SVM_%d_%s_%s' % (syllable_string,
                                                          num_mix,
                                                               name,template_tag)),)
            except:
                import pdb; pdb.set_trace()
            if verbose:
                print '%s_w_b_second_stage_SVM_%d_%s_%s' % (syllable_string,
                                                          num_mix,
                                                               name,template_tag)
            cascade_names += ('%s_w_b_second_stage_%d_%d_%s_%s' % (syllable_string,
                                                          mix_component,
                                                          num_mix,
                                                               name,template_tag),)
            if verbose:
                print '%s_w_b_second_stage_%d_%d_%s_%s' % (syllable_string,
                                                          mix_component,
                                                          num_mix,
                                                               name,template_tag)
    np.savez('%s%s_second_layer_cascade_filters_%d_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag),
             *(tuple(
                k[1]
                for k in cascade_layer)))
    if verbose:
        print '%s%s_second_layer_cascade_filters_%d_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag)
    np.save('%s%s_second_layer_cascade_mix_components_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag),
             np.array(
                [k[0]
                for k in cascade_layer]))
    if verbose:
        print '%s%s_second_layer_cascade_mix_components_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag)
    np.save('%s%s_second_layer_cascade_constant_terms_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag),
             np.array(
                [k[2]
                for k in cascade_layer]))
    if verbose:
        print '%s%s_second_layer_cascade_constant_terms_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag)
    np.save('%s%s_second_layer_cascade_identities_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag),
             np.array(
                [k[3]
                for k in cascade_layer]))
    if verbose:
        print '%s%s_second_layer_cascade_identities_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag)
    np.save('%s%s_second_layer_cascade_names_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag),
             cascade_names)
    if verbose:
        print '%s%s_second_layer_cascade_names_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               save_tag)


def apply_first_layer_cascade(num_mix,train_example_lengths,
                         train_path,file_indices,syllable,sp,
                         ep,leehon_mapping,
                         pp=None,
                         save_tag='',template_tag=None,savedir='data/',verbose=False,num_use_file_idx=-1):

    detection_array = np.zeros((train_example_lengths.shape[0],
                            train_example_lengths.max() + 2),dtype=np.float32)

    outfile = np.load('%s%s_first_layer_cascade_%d_%s.npz' % (savedir,
                                                    syllable_string,
                                                    num_mix,
                                                    template_tag))
    cascade = ()
    for i in xrange(len(outfile.files)/2):
        cascade += ((outfile['arr_%d' % (2*i)],np.float32(outfile['arr_%d' % (2*i+1)])),)


    full_detection_array = np.zeros((train_example_lengths.shape[0],
                                len(cascade),
                                train_example_lengths.max() + 2),dtype=np.float32)


    if num_use_file_idx == -1:
        num_use_file_idx = len(file_indices)

    example_starts_ends = []
    detect_lengths=[]

    for fl_id, fl in enumerate(file_indices[:num_use_file_idx]):
        utterance = makeUtterance(test_path,fl_idx)
        sflts = (utterance.flts * utterance.s.shape[0]/float(utterance.flts[-1]) + .5).astype(int)
        S = get_spectrogram(utterance.s,sp)
        S_flts = utterance.flts
        E = get_edge_features(S.T,ep,verbose=False)
        E_flts = S_flts
        if pp is not None:
            # E is now the part features
            E = get_part_features(E,pp,verbose=False)
            E_flts = S_flts.copy()
            E_flts[-1] = len(E)
        (detect_length,
         utt_example_starts_ends) = gtrd.compute_detection_E(E,phns,E_flts,
                                                        detection_array[fl_id],
                                                        cascade,
                                                        syllable,
                                                        phn_mapping=None,
                                                        verbose=False)
        detect_lengths.append(detect_length)
        example_starts_ends.append(utt_example_starts_ends)
        # check whether there is a local max in the example_starts_ends



    detection_array = np.max(detection_array,axis=1)
    detection_template_ids = np.argmax(detection_array,axis=1)
    np.save('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_array)
    np.save('%sfull_detection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),full_detection_array)

    np.save('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_template_ids)
    np.save('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detect_lengths)
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'wb')
    pickle.dump(example_start_end_times,out)
    out.close()



def save_detection_setup(num_mix,train_example_lengths,
                         train_path,file_indices,syllable,sp,
                         ep,leehon_mapping,
                         pp=None,
                         save_tag='',template_tag=None,savedir='data/',verbose=False,num_use_file_idx=-1, use_svm_based=False,syllable_string=None,
                         svm_name=None,use_svm_filter=None,
                         use_noise_file=None,
                         noise_db=0,
                         use_spectral=False,
                         load_data_tag=None):
    """
    Opens:
    ======
    '%s%s_svm_based_bgd.npy' % (savedir,syllable_string)
    '%sspec_bgd_%s.npy' % (savedir,load_data_tag)
    '%sbgd_%s.npy' %(savedir,load_data_tag)


    Saves:
    ======
    '%slinear_filter_%d.npz'% (savedir,num_mix),*(tuple(lfc[0] for lfc in linear_filters_cs))
    '%sc_%d.npz'%(savedir,num_mix),*(tuple(lfc[1] for lfc in linear_filters_cs))
    '%slinear_filter_%d_%s.npz'% (savedir,num_mix,template_tag),*(tuple(lfc[0] for lfc in linear_filters_cs))
    '%sc_%d_%s.npz'%(savedir,num_mix,template_tag),*(tuple(lfc[1] for lfc in linear_filters_cs))
    '%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag)
    '%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag)
    '%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag)
    '%sexample_start_end_times_%s.pkl' % (savedir,save_tag)

    """
    if load_data_tag is None:
        load_data_tag = template_tag
    if use_svm_based and syllable_string is not None:
        bgd = np.load('%s%s_svm_based_bgd.npy' % (savedir,syllable_string))
    elif use_spectral:
        bgd = np.load('%sspec_bgd_%s.npy' % (savedir,load_data_tag))
        bgd_sigma = np.load('%sspec_bgd_sigma_%s.npy' % (savedir,load_data_tag))
    else:
        bgd = np.load('%sbgd_%s.npy' %(savedir,load_data_tag))

    try:

        out =get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                             use_svm_based=use_svm_based,
                             svm_name=svm_name,syllable_string=syllable_string,
                             use_spectral=use_spectral)
    except:
        print "It seems that the templates could not be loaded for"
        print "template_tag=%s\tnum_mix=%d" % (template_tag,num_mix)
        return
    if use_spectral:
        templates, sigmas = out
    else:
        templates = out
    detection_array = np.zeros((train_example_lengths.shape[0],
                            train_example_lengths.max() + 2),dtype=np.float32)
    if use_svm_filter is None:
        if use_spectral:
            linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd,use_spectral=use_spectral,T_sigmas=sigmas,bgd_sigma=bgd_sigma)
        else:

            linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd,use_spectral=use_spectral)

    else:
        linear_filters_cs = ()
        for mix_component in xrange(num_mix):
            outfile = np.load('%s%d%s' % (use_svm_filter[0],
                                          mix_component,
                                          use_svm_filter[1]))
            linear_filters_cs += ((outfile['w'].reshape(templates[mix_component].shape).astype(np.float32),
                                   float(outfile['b'])),)



    if template_tag is None:
        np.savez('%slinear_filter_%d.npz'% (savedir,num_mix),*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('%sc_%d.npz'%(savedir,num_mix),*(tuple(lfc[1] for lfc in linear_filters_cs)))
    else:
        np.savez('%slinear_filter_%d_%s.npz'% (savedir,num_mix,template_tag),*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('%sc_%d_%s.npz'%(savedir,num_mix,template_tag),*(tuple(lfc[1] for lfc in linear_filters_cs)))

    if num_use_file_idx == -1:
        num_use_file_idx = len(file_indices)

    (detection_array,
     example_start_end_times,
     detection_lengths,
     detection_template_ids)=gtrd.get_detection_scores_mixture_named_params(
             train_path,
             file_indices[:num_use_file_idx],
             detection_array,
             syllable,
             linear_filters_cs,S_config=sp,
             E_config=ep,
             verbose = verbose,
             num_examples =-1,
             return_detection_template_ids=True,
             phn_mapping=leehon_mapping,
             P_config=pp,
             use_noise_file=use_noise_file,
             noise_db=noise_db,
             use_spectral=use_spectral)
    np.save('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_array)
    np.save('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_template_ids)
    np.save('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),detection_lengths)

    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'wb')

    pickle.dump(example_start_end_times,out)
    out.close()


def get_classification_scores(num_mix,data_classify_lengths,
                              data_path,file_indices,sp,
                              ep,
                              pp=None,
                              save_tag='',template_tag=None,savedir='data/',verbose=False,num_use_file_idx=-1, use_svm_based=False,syllable_string=None,
                              svm_name=None,use_svm_filter=None,
                              use_noise_file=None,
                         noise_db=0,
                         use_spectral=False,
                         load_data_tag=None):
    """
    Opens:
    ======
    '%s%s_svm_based_bgd.npy' % (savedir,syllable_string)
    '%sspec_bgd_%s.npy' % (savedir,load_data_tag)
    '%sbgd_%s.npy' %(savedir,load_data_tag)


    Saves:
    ======
    '%slinear_filter_%d.npz'% (savedir,num_mix),*(tuple(lfc[0] for lfc in linear_filters_cs))
    '%sc_%d.npz'%(savedir,num_mix),*(tuple(lfc[1] for lfc in linear_filters_cs))
    '%slinear_filter_%d_%s.npz'% (savedir,num_mix,template_tag),*(tuple(lfc[0] for lfc in linear_filters_cs))
    '%sc_%d_%s.npz'%(savedir,num_mix,template_tag),*(tuple(lfc[1] for lfc in linear_filters_cs))
    '%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag)
    '%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag)
    '%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag)
    '%sexample_start_end_times_%s.pkl' % (savedir,save_tag)

    """
    if load_data_tag is None:
        load_data_tag = template_tag
    if use_svm_based and syllable_string is not None:
        bgd = np.load('%s%s_svm_based_bgd.npy' % (savedir,syllable_string))
    elif use_spectral:
        bgd = np.load('%sspec_bgd_%s.npy' % (savedir,load_data_tag))
        bgd_sigma = np.load('%sspec_bgd_sigma_%s.npy' % (savedir,load_data_tag))
    else:
        bgd = np.load('%sbgd_%s.npy' %(savedir,load_data_tag))

    try:

        out =get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                             use_svm_based=use_svm_based,
                             svm_name=svm_name,syllable_string=syllable_string,
                             use_spectral=use_spectral)
    except:
        print "It seems that the templates could not be loaded for"
        print "template_tag=%s\tnum_mix=%d" % (template_tag,num_mix)
        return
    if use_spectral:
        templates, sigmas = out
    else:
        templates = out
    classify_array = np.zeros((data_classify_lengths.shape[0],
                            data_classify_lengths.max() + 2),dtype=np.float32)
    if use_svm_filter is None:
        if use_spectral:
            linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd,use_spectral=use_spectral,T_sigmas=sigmas,bgd_sigma=bgd_sigma)
        else:

            linear_filters_cs = et.construct_linear_filters(templates,
                                                        bgd,use_spectral=use_spectral)

    else:
        linear_filters_cs = ()
        for mix_component in xrange(num_mix):
            outfile = np.load('%s%d%s' % (use_svm_filter[0],
                                          mix_component,
                                          use_svm_filter[1]))
            linear_filters_cs += ((outfile['w'].reshape(templates[mix_component].shape).astype(np.float32),
                                   float(outfile['b'])),)



    if template_tag is None:
        np.savez('%slinear_filter_%d.npz'% (savedir,num_mix),*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('%sc_%d.npz'%(savedir,num_mix),*(tuple(lfc[1] for lfc in linear_filters_cs)))
    else:
        np.savez('%slinear_filter_%d_%s.npz'% (savedir,num_mix,template_tag),*(tuple(lfc[0] for lfc in linear_filters_cs)))
        np.savez('%sc_%d_%s.npz'%(savedir,num_mix,template_tag),*(tuple(lfc[1] for lfc in linear_filters_cs)))

    if num_use_file_idx == -1:
        num_use_file_idx = len(file_indices)

    (classify_array,
     classify_locs,
     classify_template_lengths,
     classify_template_ids)=gtrd.get_classify_scores(
             data_path,
             file_indices[:num_use_file_idx],
             classify_array,
             linear_filters_cs,
             bgd.astype(np.float32),
             S_config=sp,
             E_config=ep,
             verbose = verbose,
             num_examples =-1,
             return_detection_template_ids=True,
             P_config=pp,
             use_noise_file=use_noise_file,
             noise_db=noise_db,
             use_spectral=use_spectral)
    np.save('%sclassify_array_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),classify_array)
    np.save('%sclassify_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),classify_template_ids)
    np.save('%sclassify_template_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),classify_template_lengths)
    np.save('%sclassify_locs_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),classify_locs)




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
                       get_plots=False,old_max_detect_tag='train_parts',
                       use_spectral=False,template_tag=None):
    """
    LOADS:
    ======
    '%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag)
    '%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag)
    '%sexample_start_end_times_%s.pkl' % (savedir,save_tag)

    SAVES:
    ======
    '%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_vals)
    '%smax_detect_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_ids)
    '%smax_detect_utt_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_utt_ids)


    np.save('%sfpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
            roc_out[0])
    np.save('%stpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
                roc_out[1])

    """
    if template_tag is None:
        template_tag = old_max_detect_tag
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'rb')
    example_start_end_times = pickle.load(out)
    out.close()



    template_out=get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                      use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates = template_out

    window_start = -int(np.mean(tuple( t.shape[0] for t in templates))/3.+.5)
    window_end = -window_start
    max_detect_vals,max_detect_ids,max_detect_utt_ids = rf.get_max_detection_in_syllable_windows(detection_array,
                                                                   example_start_end_times,
                                                                   detection_lengths,
                                                                   window_start,
                                                                   window_end,
                                                                              return_argsort_idx=True)
    np.save('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_vals)
    np.save('%smax_detect_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_ids)
    np.save('%smax_detect_utt_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_utt_ids)
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
        plt.savefig('%s%s_fp_roc_discriminationLike_1stage_full_%d_%s.png' % (savedir,syllable_string,

                                                            num_mix,
                                                                          save_tag))
        plt.close('all')
        use_idx = roc_out[0] <1.
        plt.plot(roc_out[0][use_idx],roc_out[1][use_idx])
        plt.ylabel('Percent True Positives Retained')
        plt.xlabel('False Positives Per Second')
        plt.title('ROC %s 1-stage Likelihood num_mix=%d' %(syllable_string,
                                                                  num_mix))
        plt.savefig('%s%s_fp_roc_discriminationLike_1stage_limited_%d_%s.png' % (savedir,syllable_string,

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


def get_fpr_tpr_classify(num_mix,data_classify_lengths,
                         phn,mapping,reject_phns,use_phns,data_type='train',
                         save_tag='',savedir='data/',
                         get_plots=False,

                         use_spectral=False,
                         template_tag=None):
    """
    Parameters:
    ===========
    data_type: str
       Should be 'train' or 'test', decides which labels to get

    LOADS:
    ======
    '%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag)
    '%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag)
    '%sexample_start_end_times_%s.pkl' % (savedir,save_tag)

    SAVES:
    ======
    '%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_vals)
    '%smax_detect_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_ids)
    '%smax_detect_utt_ids_%d_%s.npy' % (savedir,num_mix,
                                                save_tag),max_detect_utt_ids)


    np.save('%sfpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
            roc_out[0])
    np.save('%stpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
                roc_out[1])

    """
    classify_array = np.load('%sclassify_array_%d_%s.npy' % (savedir,num_mix,
                                                             save_tag))
    classify_template_lengths = np.load('%sclassify_template_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))

    classify_labels = np.load('%s%s_classify_labels.npy' % (savedir,data_type))


    template_out=get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                      use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates = template_out

    (sorted_positive_scores,
     sorted_positive_locs,
     sorted_negative_scores,
     sorted_negative_locs,
     sorted_use_phns,
     sorted_negative_counts) = rf.get_true_positive_classify_scores(phn,classify_array,classify_labels,data_classify_lengths,use_phns,mapping)
    np.savez('%ssorted_scoring_arrays_%d_%s.npz' % (savedir,num_mix,
                                                save_tag),
                                                sorted_positive_scores=sorted_positive_scores,
                                                sorted_positive_locs=sorted_positive_locs,
                                                sorted_negative_scores=sorted_negative_scores,
                                                sorted_negative_locs=sorted_negative_locs,
                                                sorted_use_phns=sorted_use_phns,
                                                sorted_negative_counts=sorted_negative_counts)

    fpr,tpr  = rf.get_classify_roc_curve(sorted_positive_scores,
                                         sorted_negative_scores,
                                         data_classify_lengths)

    np.save('%sfpr_classify_stage1_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
            fpr)

    np.save('%stpr_classify_stage1_%d_%s.npy' % (savedir,num_mix,
                                    save_tag),
                tpr)
    if get_plots:
        plt.close('all')
        plt.plot(tpr,fpr)
        plt.xlabel('Percent True Positives Retained')
        plt.ylabel('False Positives Per Second')
        plt.title('ROC %s 1-stage Likelihood num_mix=%d' %(str(phn),
                                                                  num_mix))
        plt.savefig('%s%s_fp_roc_discriminationLike_1stage_full_%d_%s.png' % (savedir,str(phn),

                                                            num_mix,
                                                                          save_tag))
        plt.close('all')

        plt.close('all')


def get_max_classification_results(num_mix,save_tag_suffix,
                                   data_type,savedir,
                                  use_phns,rejected_phns,mapping,
                                  data_classify_lengths,verbose=False):
    """
    Finds the best phone at each point

    Parameters:
    ===========
    data_type: str
        Should be 'train' or 'test' depending on which labels one is interested in
    save_tag_suffix: str
        along the lines of 'train_edges_no_smoothing' where every phn
        result file is saved as ${phn}_train_edges_no_smoothing, this
        is the suffix of the tags where the results are saved
    """
    classify_labels = np.load('%s%s_classify_labels.npy' % (savedir,data_type))
    max_classify_array=-np.inf * np.ones((data_classify_lengths.shape[0],
                                 data_classify_lengths.max()),
                                dtype=np.float32)
    # indicates which phone was the maxima at each utterance
    argmax_classify_array=np.zeros((data_classify_lengths.shape[0],
                                 data_classify_lengths.max()),
                                dtype='|S4')
    max_classify_template_lengths = np.zeros(max_classify_array.shape,dtype=np.uint16)
    max_classify_template_ids = np.zeros(max_classify_array.shape,dtype=np.uint16)
    max_classify_locs = np.zeros(max_classify_array.shape,dtype=np.uint16)

    # these copy array keep track of statistics about the ground truth
    # labeled, this assists
    true_max_classify_array=-np.inf * np.ones((data_classify_lengths.shape[0],
                                 data_classify_lengths.max()),
                                dtype=np.float32)
    true_max_classify_template_lengths = np.zeros(max_classify_array.shape,dtype=np.uint16)
    true_max_classify_template_ids = np.zeros(max_classify_array.shape,dtype=np.uint16)
    true_max_classify_locs = np.zeros(max_classify_array.shape,dtype=np.uint16)

    # use_phn_index_dict = dict( (k,v) for v,k in enumerate(use_phns))
    # argmax_classify_array_ints = -1 *np.ones(argmax_classify_array.shape,
    #                                       dtype=np.int16)
    # classify_labels_ints = argmax_classify_array_ints.copy()
    # for utt_id, utt_phns in enumerate(argmax_classify_array_ints.shape[0]):
    #     for phn_id, phn in enumerate(utt_phns):
    #         if phn in use_phn_index_dict:
    #             argmax_classify_array_ints[utt_id,phn_id] = (
    #                 use_phn_index_dict[phn])
    #         true_phn = classify_labels
    #         if true_phn in use_phn_index_dict:
    #             classify_labels_ints[utt_id,phn_id] = (
    #                 use_phn_index_dict[true_phn])


    # run through the maximization, also fill in the ground truth arrays
    for use_phn_id, use_phn in enumerate(use_phns):
        save_tag = '%s_%s' % (use_phn,save_tag_suffix)
        if verbose:
            print save_tag
        classify_array = np.load('%sclassify_array_%d_%s.npy' % (savedir,num_mix,
                                                             save_tag))[:,:data_classify_lengths.max()]
        classify_template_lengths = np.load('%sclassify_template_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))[:,:data_classify_lengths.max()]
        classify_template_ids = np.load('%sclassify_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))[:,:data_classify_lengths.max()]
        classify_locs = np.load('%sclassify_locs_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))[:,:data_classify_lengths.max()]

        # peform the maximization
        improved_entries = classify_array > max_classify_array
        max_classify_array[improved_entries] = classify_array[improved_entries]
        if not np.all((max_classify_array - classify_array)[improved_entries] == 0):
            import pdb; pdb.set_trace()
        argmax_classify_array[improved_entries] = use_phn
        max_classify_template_lengths[improved_entries] = classify_template_lengths[improved_entries]
        max_classify_template_ids[improved_entries] = classify_template_ids[improved_entries]
        max_classify_locs[improved_entries] = classify_locs[improved_entries]

        # record the ground truth statistics
        true_entries = classify_labels == use_phn
        true_max_classify_array[true_entries] = classify_array[true_entries]
        true_max_classify_template_lengths[true_entries] = classify_template_lengths[true_entries]
        true_max_classify_template_ids[true_entries] = classify_template_ids[true_entries]
        true_max_classify_locs[true_entries] = classify_locs[true_entries]

    np.savez('%smax_classify_results_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag_suffix),
             max_classify_array=max_classify_array,
             max_classify_template_ids=max_classify_template_ids,
             max_classify_template_lengths=max_classify_template_lengths,
             max_classify_locs=max_classify_locs,
             argmax_classify_array=argmax_classify_array)

    np.savez('%strue_max_classify_results_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag_suffix),
             true_max_classify_array=true_max_classify_array,
             true_max_classify_template_ids=true_max_classify_template_ids,
             true_max_classify_template_lengths=true_max_classify_template_lengths,
             true_max_classify_locs=true_max_classify_locs)


def get_classify_confusion_matrix(num_mix,save_tag,savedir,data_type,
                                  use_phns,top_confusions=5000,verbose=False):
    """
    Compute the confusion matrix

    Parameters:
    ===========
    data_type: str
        Should be 'train' or 'test' depending on which labels one is interested in
    """
    classify_labels = np.load('%s%s_classify_labels.npy' % (savedir,data_type))
    outfile = np.load('%smax_classify_results_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag))
    argmax_classify_array = outfile['argmax_classify_array']
    max_classify_template_ids = outfile['max_classify_template_ids']
    outfile = np.load('%strue_max_classify_results_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag))
    true_max_classify_array = outfile['true_max_classify_array']
    true_max_classify_template_ids = outfile['true_max_classify_template_ids']
    num_use_phns = len(use_phns)
    # confusion_matrix[i,j] is the number of times phn i was the max but phn j was the true label
    # it is inherently asymmetric because there are two types of errors
    confusion_matrix = np.zeros((num_use_phns,num_use_phns),dtype=np.uint16)
    confusion_matrix_by_id = np.zeros((num_use_phns*num_mix,
                                       num_use_phns*num_mix),dtype=np.uint16)
    for use_phns_id_classified, use_phn_classified in enumerate(use_phns):
        if verbose:
            print use_phns_id_classified, use_phn_classified

        classified_mask = argmax_classify_array == use_phn_classified
        true_labels_for_classified = classify_labels[classified_mask]
        for use_phns_id_truth, use_phn_truth in enumerate(use_phns):
            truth_for_classified_mistakes = np.logical_and(classified_mask,classify_labels == use_phn_truth)
            confusion_matrix[use_phns_id_classified,
                             use_phns_id_truth] = np.sum(
                truth_for_classified_mistakes)
            for mix_component_classified in xrange(num_mix):
                try:
                    truth_for_classified_mistakes_mix_component =np.logical_and(truth_for_classified_mistakes,
                                                                            max_classify_template_ids == mix_component_classified)
                except:
                    import pdb; pdb.set_trace()

                for mix_component_truth in xrange(num_mix):
                    confusion_matrix_by_id[use_phns_id_classified*num_mix
                                           +mix_component_classified,
                                           use_phns_id_truth*num_mix
                                           +mix_component_truth] = np.sum(
                        np.logical_and(truth_for_classified_mistakes_mix_component,
                                       true_max_classify_template_ids == mix_component_truth))


    np.savez('%sclassify_confusion_mat_use_phns_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag),
             confusion_matrix=confusion_matrix,
             confusion_matrix_by_id=confusion_matrix_by_id,
             use_phns=use_phns)


def get_classify_scores_metadata(num_mix,phn,save_tag,savedir,
                                      data_type,use_phns,classify_lengths,
                                      verbose=False):
    if verbose:
        print "doing get_classify_scores_metadata for %s" % phn
    classify_labels = np.load('%s%s_classify_labels.npy' % (savedir,data_type))

    classify_lengths = classify_lengths.astype(np.uint16)

    outfile = np.load('%smax_classify_results_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag))
    argmax_classify_array = outfile['argmax_classify_array']
    max_classify_array=outfile['max_classify_array'].astype(np.float32)
    max_classify_template_ids = outfile['max_classify_template_ids']
    max_classify_template_lengths = outfile['max_classify_template_lengths']
    max_classify_locs = outfile['max_classify_locs'].astype(np.uint16)
    outfile = np.load('%strue_max_classify_results_%d_%s.npz' % (savedir,
                                               num_mix,
                                               save_tag))
    true_max_classify_array = outfile['true_max_classify_array'].astype(np.float32)
    true_max_classify_locs = outfile['true_max_classify_locs'].astype(np.uint16)

    true_max_classify_template_ids = outfile['true_max_classify_template_ids']
    num_use_phns = len(use_phns)

    # get the false positives
    mistake_mask = np.logical_and(argmax_classify_array == phn,
                              classify_labels != phn)
    num_mistakes = np.int(mistake_mask.sum())
    num_mistakes_by_component = np.zeros(num_mix)
    mistake_lengths = np.zeros(num_mix)

    for mix_component in xrange(num_mix):
        component_mistake_mask = (np.logical_and(mistake_mask,
                                                 max_classify_template_ids==mix_component))
        num_mistakes_by_component[mix_component] =  component_mistake_mask.sum()

        if num_mistakes_by_component[mix_component] == 0: continue

        mistake_lengths[mix_component] = max_classify_template_lengths[component_mistake_mask][0]
        component_mistake_mask=component_mistake_mask.astype(np.uint8)

        # get the length



        mistake_scores, mistake_metadata = get_mistakes.get_example_scores_metadata(component_mistake_mask,
                                                                     max_classify_locs,
                                                                     max_classify_array,
                                                                     classify_lengths,
                                                                     num_mistakes_by_component[mix_component])

        sorted_mistake_ids = np.argsort(mistake_scores)[::-1]
        np.savez('%sstage1_mistake_scores_metadata_%d_%d_%s_%s' %(
                savedir,

                num_mix,
                mix_component,phn,
                save_tag),
                 mistake_scores=mistake_scores,
                 mistake_metadata=mistake_metadata,
                 mistake_lengths=mistake_lengths[mix_component])

    # get the false negatives
    false_neg_mask = np.logical_and(argmax_classify_array != phn,
                              classify_labels == phn)
    num_false_negs = np.int(false_neg_mask.sum())
    num_false_negs_by_component = np.zeros(num_mix)
    false_neg_lengths = np.zeros(num_mix)
    for mix_component in xrange(num_mix):
        component_false_neg_mask = (np.logical_and(false_neg_mask,
                                                 max_classify_template_ids==mix_component))
        num_false_negs_by_component[mix_component] =  component_false_neg_mask.sum()

        if num_false_negs_by_component[mix_component] == 0: continue
        false_neg_lengths[mix_component] = max_classify_template_lengths[component_false_neg_mask][0]
        component_false_neg_mask = component_false_neg_mask.astype(np.uint8)
        # get the length

        false_neg_scores, false_neg_metadata = get_mistakes.get_example_scores_metadata(component_false_neg_mask,
                                                                     true_max_classify_locs,
                                                                     true_max_classify_array,
                                                                     classify_lengths,
                                                                     num_false_negs_by_component[mix_component])

        sorted_false_neg_ids = np.argsort(false_neg_scores)[::-1]
        np.savez('%sstage1_false_neg_scores_metadata_%d_%d_%s_%s' %(
                savedir,

                num_mix,
                mix_component,phn,
                save_tag),
                 false_neg_scores=false_neg_scores,
                 false_neg_metadata=false_neg_metadata,
                 false_neg_lengths=false_neg_lengths[mix_component])




    # get the true positives
    success_mask = np.logical_and(argmax_classify_array == phn,
                              classify_labels == phn)
    num_successes = np.int(success_mask.sum())
    num_successes_by_component = np.zeros(num_mix)
    success_lengths = np.zeros(num_mix)
    for mix_component in xrange(num_mix):
        component_success_mask = (np.logical_and(success_mask,
                                                 max_classify_template_ids==mix_component))
        num_successes_by_component[mix_component] =  component_success_mask.sum()

        if num_successes_by_component[mix_component] == 0: continue
        success_lengths[mix_component] = max_classify_template_lengths[component_success_mask][0]
        component_success_mask=component_success_mask.astype(np.uint8)
        # get the length

        success_scores, success_metadata = get_mistakes.get_example_scores_metadata(component_success_mask,
                                                                     max_classify_locs,
                                                                     max_classify_array,
                                                                     classify_lengths,
                                                                     num_successes_by_component[mix_component])

        sorted_success_ids = np.argsort(success_scores)[::-1]
        np.savez('%sstage1_success_scores_metadata_%d_%d_%s_%s' %(
                savedir,

                num_mix,
                mix_component, phn,
                save_tag),
                 success_scores=success_scores,
                 success_metadata=success_metadata,
                 success_lengths=success_lengths[mix_component])






def get_tagged_all_detection_clusters(num_mix,save_tag,old_max_detect_tag,savedir,use_spectral=False):
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,
                                                            num_mix,save_tag))
    max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,
                                                               num_mix,
                                                               old_max_detect_tag))
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,
                                                                   num_mix,
                                                                   save_tag))

    template_out=get_templates(num_mix,template_tag=old_max_detect_tag,savedir=savedir,
                      use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates = template_out

    C0 = int(np.mean(tuple( t.shape[0] for t in templates))+.5)
    C1 = int( 33 * 1.5 + .5)
    detection_clusters = rf.get_detect_clusters_threshold_array(
        max_detect_vals,
        detection_array,
        detection_lengths,
        C0,C1)
    out = open('%sdetection_clusters_%d_%s.pkl' %(savedir,num_mix,
                                                  save_tag),'wb')
    cPickle.dump(detection_clusters,out)
    out.close()
    return detection_clusters

def get_tagged_detection_clusters(num_mix,thresh_percent,save_tag='',use_thresh=None,old_max_detect_tag=None,savedir='data/',use_spectral=False,template_tag=None):
    """
    Loads:
    ======
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))
    max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,old_max_detect_tag))
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))
    template_out=get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                      use_spectral=use_spectral)

    detection_template_ids = np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                    save_tag))



    Saves:
    ======
    out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (savedir,num_mix,
                                                                       thresh_percent,
                                                                       save_tag)
                                                                    ,'wb')
    """
    if template_tag is None:
        template_tag=old_max_detect_tag
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))
    if old_max_detect_tag is None:
        max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,save_tag))
    else:
        max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,old_max_detect_tag))

    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                  save_tag))

    template_out=get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                      use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates = template_out

    C0 = int(np.mean(tuple( t.shape[0] for t in templates))+.5)
    C1 = int( 33 * 1.5 + .5)

    thresh_id = max(int((len(max_detect_vals)-1) *thresh_percent/float(100)),
                    np.arange(len(max_detect_vals))[max_detect_vals > -np.inf][0])
    if use_thresh is None:
        detect_thresh =max_detect_vals[thresh_id]
    else:
        detect_thresh =use_thresh

    #import pdb; pdb.set_trace()

    print detect_thresh

    detection_template_ids = np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,
                                                    save_tag))

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


def cluster_true_pos_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',savedir='data/',template_tag='train_parts',
                               verbose=False,pp=None,num_use_file_idx=None,
                               max_num_points_cluster=1000,
                              use_spectral=False):
    if thresh_percent is None and save_tag=='':
        out = open('%s%s_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')
        if verbose:
            print "using file %s%s_true_pos_times_%d_%d_%s.pkl for true_positive hunting" % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag)
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

    true_pos_ids = np.array(tuple(
        fpd.cluster_detect_ids[fpd.cluster_max_peak_loc]
        for fpd in reduce(lambda x,y: x+y,true_pos_times)))

    true_pos_thresh_list = get_thresh_list(num_mix,
                                            true_pos_scores,
                                            true_pos_ids,
                                            max_num_points_cluster)


    template_out=get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                      use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates = template_out


    if verbose:
        print "true_pos_thresh_list = %s" % str(true_pos_thresh_list)


    for mix_component in xrange(num_mix):
        print "mix_component=%d" % mix_component

        true_pos_times_component = get_thresholded_subset_detect_times_component(mix_component,
        true_pos_thresh_list,
        true_pos_times)


        out = open('%s%s_true_pos_times_%d_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,mix_component,
                                                                    thresh_percent,
                                                                    save_tag),'wb')
        pickle.dump(true_pos_times_component,out)
        out.close()



        if num_use_file_idx is None or num_use_file_idx==-1:
            num_use_file_idx = len(true_pos_times_component)
        true_positives = rf.get_true_positives(true_pos_times_component[:num_use_file_idx],
                                                 S_config=sp,
                                                 E_config=ep,
                                                 P_config=pp,
                                                 offset=0,
                                                 waveform_offset=waveform_offset,
                                                 verbose=verbose)

        example_mat = gtrd.recover_example_map(true_positives)

        np.save('%s%s_true_positives_example_mat_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),example_mat)
        lengths,waveforms  = gtrd.recover_waveforms(true_positives,example_mat)
        np.savez('%s%s_true_positives_waveforms_lengths_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                                   num_mix,mix_component,
                                                                         thresh_percent,
                                                                         save_tag),
             lengths=lengths,
             waveforms=waveforms)

        if max(len(e) for  e in true_positives) == 0:
            continue
        Slengths,Ss  = gtrd.recover_specs(true_positives,example_mat)

        if not np.all(Slengths == Ss.shape[1]):
            print "Something is wrong"
            import pdb; pdb.set_trace()

        np.save('%s%s_true_positives_Ss_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag),
             Ss)
        Elengths,Es  = gtrd.recover_edgemaps(true_positives,example_mat)
        assert np.all(Elengths == Es.shape[1])
        if verbose:
            print "num_mix=%d,mix_component=%d,len(example_mat)=%d,len(Es[0])=%d,len(templates[%d])=%d" %(num_mix,mix_component,len(example_mat),len(Es[0]),mix_component,len(templates[mix_component]))
        if verbose:
            print 'Number of true positive Es is: %d' % len(Es)
        np.save('%s%s_true_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag),
             Es)





def get_true_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',savedir='data/',
                           verbose=False,pp=None):
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
                                           P_config=pp,
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
                           verbose=False,pp=None,num_use_file_idx=None,
                           max_num_points_cluster=1000):
    if thresh_percent is None and save_tag=='':
        out = open('%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')
        if verbose:
            print "using file %s%s_false_pos_times_%d_%d_%s.pkl for false_positive hunting" % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag)
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

    false_pos_ids = np.array(tuple(
        fpd.cluster_detect_ids[fpd.cluster_max_peak_loc]
        for fpd in reduce(lambda x,y: x+y,false_pos_times)))

    false_pos_thresh_list = get_thresh_list(num_mix,
                                            false_pos_scores,
                                            false_pos_ids,
                                            max_num_points_cluster)


    thresholded_false_pos_times = get_thresholded_subset_detect_times(
        false_pos_thresh_list,
        false_pos_times)


    out = open('%s%s_false_pos_times_unthresholded_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                                    thresh_percent,
                                                                    save_tag),'wb')
    pickle.dump(false_pos_times,out)
    out.close()

    out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                                    thresh_percent,
                                                                    save_tag),'wb')
    pickle.dump(thresholded_false_pos_times,out)
    out.close()



    if num_use_file_idx is None or num_use_file_idx==-1:
        num_use_file_idx = len(thresholded_false_pos_times)
    false_positives = rf.get_false_positives(thresholded_false_pos_times[:num_use_file_idx],
                                             S_config=sp,
                                             E_config=ep,
                                             P_config=pp,
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
    if verbose:
        print 'Number of false positive Es is: %d' % len(Es)
    np.savez('%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag),
             lengths=Elengths,
             Es=Es)


def cluster_false_pos_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',template_tag='train_parts',savedir='data/',
                               verbose=False,pp=None,num_use_file_idx=None,
                               max_num_points_cluster=1000,
                               use_spectral=False):
    """
    Loads:
    ======
        out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')


    Saves:
    ======
        out = open('%s%s_false_pos_times_%d_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,mix_component,
                                                                    thresh_percent,
                                                                    save_tag),'wb')
        np.save('%s%s_false_positive_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),
            false_pos_scores)
        np.save('%s%s_false_positives_scores_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),false_pos_scores_component)
        np.save('%s%s_false_positives_example_mat_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),example_mat)
        np.savez('%s%s_false_positives_waveforms_lengths_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                                   num_mix,mix_component,
                                                                         thresh_percent,
                                                                         save_tag),
             lengths=lengths,
             waveforms=waveforms)

        np.save('%s%s_false_positives_Ss_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag),

             Ss)
        np.save('%s%s_false_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag),

             Es)


    """
    if thresh_percent is None and save_tag=='':
        out = open('%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag),'rb')
        if verbose:
            print "using file %s%s_false_pos_times_%d_%d_%s.pkl for false_positive hunting" % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             save_tag)
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

    false_pos_ids = np.array(tuple(
        fpd.cluster_detect_ids[fpd.cluster_max_peak_loc]
        for fpd in reduce(lambda x,y: x+y,false_pos_times)))


    # get the thresholds so that we limit the number of examples
    # that we train against with the SVM
    false_pos_thresh_list = get_thresh_list(num_mix,
                                            false_pos_scores,
                                            false_pos_ids,
                                            max_num_points_cluster)



    out=get_templates(num_mix,template_tag=template_tag,savedir=savedir,
                      use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = out
    else:
        templates = out

    if verbose:
        print "false_pos_thresh_list = %s" % str(false_pos_thresh_list)


    for mix_component in xrange(num_mix):
        if verbose:
            print "mix_component=%d,num_mix=%d" % (mix_component,num_mix)
        false_pos_times_component = get_thresholded_subset_detect_times_component(mix_component,
        false_pos_thresh_list,
        false_pos_times)


        out = open('%s%s_false_pos_times_%d_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,mix_component,
                                                                    thresh_percent,
                                                                    save_tag),'wb')
        pickle.dump(false_pos_times_component,out)
        out.close()


        print "num utterances in false_pos_times_component=%d" % sum(len(u) for u in false_pos_times_component)
        if num_use_file_idx is None or num_use_file_idx==-1:
            num_use_file_idx = len(false_pos_times_component)
        false_positives = rf.get_false_positives(false_pos_times_component[:num_use_file_idx],
                                                 S_config=sp,
                                                 E_config=ep,
                                                 P_config=pp,
                                                 offset=0,
                                                 waveform_offset=waveform_offset,
                                                 verbose=verbose,
                                                 use_spectral=use_spectral)

        false_pos_scores_component = np.array(tuple(
                fpd.cluster_max_peak_val
                for fpd in reduce(lambda x,y: x+y,false_pos_times_component)))
        np.save('%s%s_false_positives_scores_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),false_pos_scores_component)
        example_mat = gtrd.recover_example_map(false_positives)
        print "len(example_mat)=%d" % len(example_mat)
        np.save('%s%s_false_positives_example_mat_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ),example_mat)
        lengths,waveforms  = gtrd.recover_waveforms(false_positives,example_mat)

        np.savez('%s%s_false_positives_waveforms_lengths_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                                   num_mix,mix_component,
                                                                         thresh_percent,
                                                                         save_tag),
             lengths=lengths,
             waveforms=waveforms)

        if max( len(e) for e in false_positives) ==0:
            continue
        else:
            Slengths,Ss  = gtrd.recover_specs(false_positives,example_mat)

        assert np.all(Slengths==Ss.shape[1])
        np.save('%s%s_false_positives_Ss_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag),

             Ss)
        Elengths,Es  = gtrd.recover_edgemaps(false_positives,example_mat)
        if not np.all(Elengths==Es.shape[1]):
            import pdb; pdb.set_trace()

        if verbose:
            print "num_mix=%d,mix_component=%d,len(example_mat)=%d,len(Es[0])=%d,len(templates[%d])=%d" %(num_mix,mix_component,len(example_mat),len(Es[0]),mix_component,len(templates[mix_component]))
        if verbose:
            print 'Number of false positive Es is: %d' % len(Es)
        np.save('%s%s_false_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,mix_component,
                                                                  thresh_percent,
                                                                  save_tag),

             Es)


def false_pos_examples_cascade_score(num_mix,syllable_string,
                                     sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',template_tag='train_parts',savedir='data/',
                               verbose=False,pp=None,num_use_file_idx=None,
                               max_num_points_cluster=1000,
                                     do_spectral_detection=False,
                                     num_extract_top_false_positives=0,
                                     old_max_detect_tag=None,
                                     load_data_tag=None):
    """
    Parameters:
    ===========
    num_extract_top_false_positives: int
        Should be non-negative, if its zero then nothing extra is done,
        otherwise we save the top `num_extract_top_false_positives`
        from each classifier, i.e. these are the hardest false positives
        in the class, we save their spectral features and their waveforms
        for later listening

    old_max_detect_tag: str
        This is the tag to identify which of the old_max_detect_values
        to use, these are the thresholds that are used from training
        to generate the ROC curves.  This is useful because we might
        have a template that has a tag associated with limited training
        data at the SVM level but not at the template estimation level

    load_data_tag: str
        This tag is used to pick the false positives to open and to
        use for processing, this is useful if the false positive set
        was generated with a template but we're doing the cascade
        with a separate function

    Saves:
    ======
    '%s%s_false_positive_cascade2_scores_%s.npy' %(
                savedir,
                k,save_tag)
        Has just all the scores, gets things ready for scoring

    '%s%s_false_positive_cascade2_component_scores_%d_%s.npy' %(
                    savedir,
                    k,mix_component,save_tag)
        component scores separate

    '%sfp_wave_%s_%d_%d_%f.wav' % (savedir,
                                                             k,
                                                             mix_component,
                                                             score_rank,
                                                             component_scores[top_score_idx])
        Optional wave writing
    '%sfp_wave_%s_%d_%d_%f.wav' % (savedir,
                                                             k,
                                                             mix_component,
                                                             score_rank,
                                                             component_scores[top_score_idx])
        optional names of the wave and Ss files
    '%sfp_S_s_%s_%d_%d_%f.npz' % (savedir,
                                                        k,
                                                        mix_component,
                                                        score_rank,
                                                        component_scores[top_score_idx])
        optional S and s from the top false positives
    """
    if old_max_detect_tag is None:
        old_max_detect_tag=template_tag
    if load_data_tag is None:
        load_data_tag = save_tag
    if thresh_percent is None and load_data_tag=='':
        out = open('%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             load_data_tag),'rb')
        if verbose:
            print "using file %s%s_false_pos_times_%d_%d_%s.pkl for false_positive hunting" % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             load_data_tag)
    false_pos_times=pickle.load(out)
    out.close()

    max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,old_max_detect_tag))
    thresh_id = int((len(max_detect_vals)-1) *thresh_percent/float(100))

    thresh_val = max_detect_vals[thresh_id]

    outfile = np.load('%s%s_second_layer_cascade_filters_%d_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    cascade_filters = tuple(outfile['arr_%d' % i] for i in xrange(len(outfile.files)))
    cascade_mix_components = np.load('%s%s_second_layer_cascade_mix_components_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    cascade_constants = np.load('%s%s_second_layer_cascade_constant_terms_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    cascade_names = np.load('%s%s_second_layer_cascade_names_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))

    cascade_identities = np.load('%s%s_second_layer_cascade_identities_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))

    false_pos_score_dict = {}
    false_pos_example_info_dict = {}

    file_idx = '-1'
    for utt_id, utt_fps in enumerate(false_pos_times):
        if len(utt_fps) ==0 : continue
        utt = gtrd.makeUtterance(utt_fps[0].utterances_path,
                                     utt_fps[0].file_index)
        S = gtrd.get_spectrogram(utt.s,sp)
        sflts = (utt.flts * utt.s.shape[0]/float(utt.flts[-1]) + .5).astype(int)

        E = gtrd.get_edge_features(S.T,ep,verbose=False
                                       )
        if pp is not None:
            # E is now the part features
            E = gtrd.get_part_features(E,pp,verbose=False)

        if verbose:
            print "working on utterance %d with file id %s" % (utt_id,
                                                               utt_fps[0].file_index)
        for fp_id, fp in enumerate(utt_fps):
        # compute cascade score:
            detect_time = fp.cluster_max_peak_loc+fp.cluster_start_end[0]
            mix_component = fp.cluster_detect_ids[fp.cluster_max_peak_loc]
            detect_length = fp.cluster_detect_lengths[fp.cluster_max_peak_loc]
            s_start = int(detect_time *sflts[-1]/float(S.shape[0]) + .5)
            s_start_end = np.array([s_start,
                                    s_start+ int(
                        detect_length * sflts[-1]/float(S.shape[0])+.5)])
            if do_spectral_detection:
                fp_features = S[detect_time:detect_time+detect_length]
            else:
                fp_features = E[detect_time:detect_time+detect_length]
            if fp_features.shape[0] < detect_length:
                fp_features = np.vstack((fp_features,
                                         np.zeros(
                            (detect_length-fp_features.shape[0],)+
                             fp_features.shape[1:])))
            for cascade_filter, cascade_constant, cascade_mix_component, cascade_id in itertools.izip(cascade_filters,cascade_constants,cascade_mix_components,cascade_identities):
                if cascade_mix_component != mix_component: continue
                else: pass

                if detect_length != cascade_filter.shape[0]:
                    import pdb; pdb.set_trace()
                assert detect_length == cascade_filter.shape[0]

                if cascade_id not in false_pos_score_dict.keys():
                    false_pos_score_dict[cascade_id] = [[] for i in xrange(num_mix)]

                if cascade_id not in false_pos_example_info_dict.keys():
                    false_pos_example_info_dict[cascade_id] = [[] for i in xrange(num_mix)]

                try:
                    false_pos_score_dict[cascade_id][cascade_mix_component].append(np.sum(fp_features * cascade_filter)+cascade_constant)
                    # saves the saved location of the file
                    # and the start end times for the s and S/E features
                    false_pos_example_info_dict[cascade_id][cascade_mix_component].append((utt_fps[0].utterances_path,
                                     utt_fps[0].file_index,
                                                                                           s_start_end,
                                                                                           (detect_time,detect_time+detect_length)))


                 #   if verbose:
                 #       print false_pos_score_dict[cascade_id][cascade_mix_component][-1]
                except:

                    import pdb; pdb.set_trace()

    if num_extract_top_false_positives>0:
        Ss_fp_file_names = ()
        Wav_fp_file_names = ()

    for k,v in false_pos_score_dict.items():
        np.save('%s%s_false_positive_cascade2_scores_%s.npy' %(
                savedir,
                k,save_tag),
                np.array(reduce(lambda x,y: x+y,
                                v)))
        for mix_component, component_scores in enumerate(v):
            np.save('%s%s_false_positive_cascade2_component_scores_%d_%s.npy' % (
                    savedir,
                    k,mix_component,save_tag),
                    component_scores)

            if num_extract_top_false_positives > 0:
                # the top scores
                top_scores = np.argsort(component_scores)[-num_extract_top_false_positives:][::-1]
                for score_rank, top_score_idx in enumerate(top_scores):
                    utt_path, fl_idx,s_start_end,S_start_end = false_pos_example_info_dict[k][mix_component][top_score_idx]
                    utt = gtrd.makeUtterance(utt_path,fl_idx)
                    S = gtrd.get_spectrogram(utt.s,sp)
                    std_s = np.std(utt.s)
                    mean_s = np.mean(utt.s)
                    top_wavechunk = np.hstack((np.random.randn(3*sp.num_window_samples)*std_s+mean_s,
                                               utt.s[max(0,s_start_end[0]-2*sp.num_window_samples):
                                          min(len(utt.s),
                                              s_start_end[1]+2*sp.num_window_samples)],
                                               np.random.randn(3*sp.num_window_samples)*std_s+mean_s))
                    top_wavechunk[:5*sp.num_window_samples] *= (np.exp(-np.arange(5*sp.num_window_samples)**2))[::-1]
                    top_wavechunk[-5*sp.num_window_samples:] *= (np.exp(-np.arange(5*sp.num_window_samples)**2))[::-1]
                    wavfile.write('%sfp_wave_%s_%d_%d_%f_%s.wav' % (savedir,
                                                             k,
                                                             mix_component,
                                                             score_rank,
                                                             component_scores[top_score_idx],
                                                                    save_tag),16000,((2**15-1)*top_wavechunk).astype(np.int16))

                    Wav_fp_file_names += ('%sfp_wave_%s_%d_%d_%f_%s.wav' % (savedir,
                                                             k,
                                                             mix_component,
                                                             score_rank,
                                                             component_scores[top_score_idx],
                                                                            save_tag),)

                    S_chunk = S[S_start_end[0]:
                                    S_start_end[1]]
                    np.savez('%sfp_S_s_%s_%d_%d_%f_%s.npz' % (savedir,
                                                        k,
                                                        mix_component,
                                                        score_rank,
                                                        component_scores[top_score_idx],save_tag),
                            S_chunk=S_chunk,
                             s_chunk=top_wavechunk)
                    Ss_fp_file_names += ('%sfp_S_s_%s_%d_%d_%f_%s.npz' % (savedir,
                                                        k,
                                                        mix_component,
                                                        score_rank,
                                                        component_scores[top_score_idx],save_tag),)

            if num_extract_top_false_positives > 0:
                np.save('%sfp_extracted_top_false_positive_fnames_Ss%s_%s.npy' % (savedir,k,save_tag),
                Ss_fp_file_names)
                np.save('%sfp_extracted_top_false_positive_fnames_Wav%s_%s.npy' % (savedir,k,save_tag),
                Wav_fp_file_names)



def true_pos_examples_cascade_score(num_mix,syllable_string,
                                     sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',template_tag='train_parts',savedir='data/',
                               verbose=False,pp=None,num_use_file_idx=None,
                               max_num_points_cluster=1000,
                                    do_spectral_detection=False,
                                    old_max_detect_tag=None,
                                    load_data_tag=None):
    """
    Parameters:
    ===========
    num_extract_bottom_true_positives: int
        Should be non-negative, if its zero then nothing extra is done,
        otherwise we save the top `num_extract_top_false_positives`
        from each classifier, i.e. these are the hardest false positives
        in the class, we save their spectral features and their waveforms
        for later listening

    old_max_detect_tag: str
        This is the tag to identify which of the old_max_detect_values
        to use, these are the thresholds that are used from training
        to generate the ROC curves.  This is useful because we might
        have a template that has a tag associated with limited training
        data at the SVM level but not at the template estimation level

    load_data_tag: str
        This tag is used to pick the false positives to open and to
        use for processing, this is useful if the false positive set
        was generated with a template but we're doing the cascade
        with a separate function
    """
    if old_max_detect_tag is None:
        old_max_detect_tag=template_tag
    if load_data_tag is None:
        load_data_tag = save_tag

    if thresh_percent is None and load_data_tag=='':
        out = open('%s%s_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             load_data_tag),'rb')
        if verbose:
            print "using file %s%s_pos_times_%d_%d_%s.pkl for true_positive hunting" % (savedir,syllable_string,num_mix,
                                                       thresh_percent,
                                                             load_data_tag)
    true_pos_times=pickle.load(out)
    out.close()

    max_detect_vals = np.load('%smax_detect_vals_%d_%s.npy' % (savedir,num_mix,old_max_detect_tag))
    thresh_id = int((len(max_detect_vals)-1) *thresh_percent/float(100))

    thresh_val = max_detect_vals[thresh_id]

    outfile = np.load('%s%s_second_layer_cascade_filters_%d_%s.npz' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    cascade_filters = tuple(outfile['arr_%d' % i] for i in xrange(len(outfile.files)))
    cascade_mix_components = np.load('%s%s_second_layer_cascade_mix_components_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    cascade_constants = np.load('%s%s_second_layer_cascade_constant_terms_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    cascade_names = np.load('%s%s_second_layer_cascade_names_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))

    cascade_identities = np.load('%s%s_second_layer_cascade_identities_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))

    true_pos_score_dict = {}
    true_pos_lengths_dict = {}
    file_idx = '-1'
    for utt_id, utt_tps in enumerate(true_pos_times):
        if len(utt_tps) ==0 : continue
        utt = gtrd.makeUtterance(utt_tps[0].utterances_path,
                                     utt_tps[0].file_index)
        S = gtrd.get_spectrogram(utt.s,sp)
        E = gtrd.get_edge_features(S.T,ep,verbose=False
                                       )
        if pp is not None:
            # E is now the part features
            E = gtrd.get_part_features(E,pp,verbose=False)

        if verbose:
            print "working on utterance %d with file id %s" % (utt_id,
                                                               utt_tps[0].file_index)
        for tp_id, tp in enumerate(utt_tps):
        # compute cascade score:
            detect_time = tp.cluster_max_peak_loc+tp.cluster_start_end[0]
            mix_component = tp.cluster_detect_ids[tp.cluster_max_peak_loc]
            detect_length = tp.cluster_detect_lengths[tp.cluster_max_peak_loc]

            if do_spectral_detection:
                tp_features = S[detect_time:detect_time+detect_length]
            else:
                tp_features = E[detect_time:detect_time+detect_length]
            if tp_features.shape[0] < detect_length:
                tp_features = np.vstack((tp_features,
                                         np.zeros(
                            (detect_length-tp_features.shape[0],)+
                             tp_features.shape[1:])))
            for cascade_filter, cascade_constant, cascade_mix_component, cascade_id in itertools.izip(cascade_filters,cascade_constants,cascade_mix_components,cascade_identities):
                if cascade_mix_component != mix_component: continue
                else: pass

                assert detect_length == cascade_filter.shape[0]

                if cascade_id not in true_pos_score_dict.keys():
                    true_pos_score_dict[cascade_id] = [[] for i in xrange(num_mix)]

                if cascade_id not in true_pos_lengths_dict.keys():
                    true_pos_lengths_dict[cascade_id] = [[] for i in xrange(num_mix)]

                try:
                    true_pos_score_dict[cascade_id][cascade_mix_component].append(np.sum(tp_features * cascade_filter)+cascade_constant)
                    true_pos_lengths_dict[cascade_id][cascade_mix_component].append(tp.true_label_times[1]-tp.true_label_times[0])
 #                   if verbose:
 #                       print true_pos_score_dict[cascade_id][cascade_mix_component][-1]

                except:

                    import pdb; pdb.set_trace()


    for k,v in true_pos_score_dict.items():
        np.save('%s%s_true_positive_cascade2_scores_%s.npy' %(
                savedir,
                k,save_tag),
                np.array(reduce(lambda x,y: x+y,
                                v)))
        for mix_component, component_scores in enumerate(v):
            np.save('%s%s_true_positive_cascade2_component_scores_%d_%s.npy' %(
                    savedir,
                    k,mix_component,save_tag),
                    component_scores)

    for k,v in true_pos_lengths_dict.items():
        np.save('%s%s_true_positive_cascade2_lengths_%s.npy' %(
                savedir,
                k,save_tag),
                np.array(reduce(lambda x,y: x+y,
                                v)))
        for mix_component, component_lengths in enumerate(v):
            np.save('%s%s_true_positive_cascade2_component_lengths_%d_%s.npy' %(
                    savedir,
                    k,mix_component,save_tag),
                    component_lengths)


def compare_cascade_scores():
    pass


def get_thresh_list(num_mix,detect_scores, detect_ids,
                    max_num_points_cluster):
    thresh_list = ()
    for i in xrange(num_mix):
        scores = np.sort(tuple(set(detect_scores[detect_ids ==i])))[::-1][:max_num_points_cluster][::-1]
        for score in scores:
            if np.sum(detect_scores[detect_ids==i] >= score) <= max_num_points_cluster:
                thresh_list += (score,)
                break

    return thresh_list

def get_thresholded_subset_detect_times(
        thresh_list,
        detect_times):

    return tuple(
        tuple(
            fpd
            for fpd in utt
            if fpd.cluster_max_peak_val >= thresh_list[
                fpd.cluster_detect_ids[fpd.cluster_max_peak_loc]])
        for utt in detect_times)

def get_thresholded_subset_detect_times_component(mix_component,
                                                  thresh_list,
                                                  detect_times,
                                                  fraction_for_window=.3):

    out_tuple = ()
    for utt in detect_times:
        out_fpds = ()
        for fpd in utt:
            try:
                if (fpd.cluster_max_peak_val >= thresh_list[
                        min(fpd.cluster_detect_ids[min(fpd.cluster_max_peak_loc,
                                                   len(fpd.cluster_detect_ids)-1)],len(thresh_list)-1)]
                    and fpd.cluster_detect_ids[fpd.cluster_max_peak_loc] == mix_component):
                    #print fpd.cluster_detect_ids[fpd.cluster_max_peak_loc]
                    #print fpd.cluster_detect_ids
                    out_fpds += (fpd,)
            except:
                import pdb;pdb.set_trace()
        out_tuple += (out_fpds,)

    return out_tuple

    try:
        return tuple(
            tuple(
                fpd
                for fpd in utt
                if (fpd.cluster_max_peak_val >= thresh_list[
                        fpd.cluster_detect_ids[fpd.cluster_max_peak_loc]]
                    and fpd.cluster_detect_ids[fpd.cluster_max_peak_loc] == mix_component))
            for utt in detect_times)
    except:
        import pdb; pdb.set_trace()






def get_false_neg_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,thresh_percent=None,save_tag='',savedir='data/',
                           verbose=False,pp=None):
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
                                             P_config=pp,
                                             offset=0,
                                             waveform_offset=waveform_offset,
                                             verbose=verbose)
    if sum(len(k) for k in false_negatives) == 0:
        return
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
                                    return_example_types=False, old_max_detect_tag='train_2',template_tag=None,

                                    detect_clusters=None,
                                    use_spectral=False):
    """
    Parameters:
    ===========
    detect_clusters:
        These are loaded as follows:
        out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (args.savedir,num_mix,
                                                                                args.thresh_percent,
                                                                                args.save_tag)


    Loads:
    ======
        out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (savedir,num_mix,
        thresh_percent,
        save_tag),'rb')
        ,'rb')
        detection_clusters =cPickle.load(out)
        detection_template_ids = np.load('%sdetection_template_ids_%d_%s.npy' % (savedir,num_mix,save_tag))
        detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                                      save_tag))
        detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                          save_tag))
        out = open('%sexample_start_end_times_%s.pkl' %(savedir,save_tag),'rb')

        example_start_end_times = pickle.load(out)


    Saves:
    ======
    """
    if template_tag is None:
        template_tag = old_max_detect_tag
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

    template_out = get_templates(num_mix,template_tag=template_tag,savedir=savedir, use_spectral=use_spectral)
    if use_spectral:

        templates,sigmas = template_out
    else:
        templates=template_out
    template_lengths = np.array([len(t) for t in templates])

    print "template_lengths=%s" % str(template_lengths)
    print "np.min(detection_template_ids)=%d, detection_template_ids.max()=%d" % (detection_template_ids.min(),detection_template_ids.max())
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

def perform_second_stage_detection_testing(num_mix,syllable_string,save_tag,thresh_percent,savedir,
                                           make_plots=False,verbose=False,old_max_detect_tag=None,
                                           use_spectral=False):
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
    #import pdb; pdb.set_trace()

    template_out = get_templates(num_mix,template_tag=old_max_detect_tag,savedir=savedir, use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates=template_out

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
    out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),'rb')
    true_pos_times=pickle.load(out)
    out.close()

    template_ids = rf.recover_template_ids_detect_times(true_pos_times)

    outfile = np.load('%s%s_true_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_true_pos = outfile['lengths']
    Es_true_pos = outfile['Es']
    outfile = np.load('%s%s_true_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_S_true_pos = outfile['lengths']
    Ss_true_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts
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
    np.savez('%s%s_clusters_for_testing_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),
             *clusters_for_classification)

    labels_for_classification = tuple(
        np.hstack((np.ones(Es_true_cluster.shape[0]),
                   np.zeros(Es_false_cluster.shape[0])))
        for Es_true_cluster, Es_false_cluster in itertools.izip(
            Es_true_pos_clusters,
            Es_false_pos_clusters))

    np.savez('%s%s_labels_for_testing_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),
             *labels_for_classification)


def perform_clustered_second_stage_detection_testing(num_mix,syllable_string,save_tag,thresh_percent,savedir,
                                           make_plots=False,verbose=False,old_max_detect_tag=None):
    templates = get_templates(num_mix,template_tag=old_max_detect_tag,savedir=savedir)
    for mix_component in xrange(num_mix):
        Es_false_pos_cluster = np.load('%s%s_false_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))
        Es_true_pos_cluster = np.load('%s%s_true_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))

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
    out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),'rb')
    true_pos_times=pickle.load(out)
    out.close()

    template_ids = rf.recover_template_ids_detect_times(true_pos_times)

    outfile = np.load('%s%s_true_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_true_pos = outfile['lengths']
    Es_true_pos = outfile['Es']
    outfile = np.load('%s%s_true_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_S_true_pos = outfile['lengths']
    Ss_true_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts
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
    np.savez('%s%s_clusters_for_testing_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),
             *clusters_for_classification)

    labels_for_classification = tuple(
        np.hstack((np.ones(Es_true_cluster.shape[0]),
                   np.zeros(Es_false_cluster.shape[0])))
        for Es_true_cluster, Es_false_cluster in itertools.izip(
            Es_true_pos_clusters,
            Es_false_pos_clusters))

    np.savez('%s%s_labels_for_testing_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),
             *labels_for_classification)



def test_predictors_second_stage(num_mix,linear_filters,Es_clusters,
                                 labels_clusters,num_time_points,num_false_negs,savedir,roc_fname,make_plots=False,verbose=False,
                                 cs=None):
    """
    Parameters:
    ===========
    num_mix: int
        Number of mixture components, corresponds to how finely the space is chopped up
    linear_filters: tuple
        Should be a tuple of length num_mix where each entry is a linear predictor which can be used on the data examples (length here of each constituent predictor should correspond to the length of the corresponding cluster)
    Es_clusters: tuple
        Again should be a tuple of length num_mix, these are the clusters
        that we are going to perform detection on using the linear_fitlers
    labels_clusters: tuple
        Another tuple of length num_mix, these tuples contain the labels
    """
    # need a way to set all of these detector thresholds
    tp_scores_idx = np.hstack(([0],np.cumsum(tuple(np.sum(labels_clusters[i] > .5) for i in xrange(num_mix)))))
    fp_scores_idx = np.hstack(([0],np.cumsum(tuple(np.sum(labels_clusters[i] < .5) for i in xrange(num_mix)))))
    tp_scores = np.empty(tp_scores_idx[-1])
    fp_scores =np.empty(fp_scores_idx[-1])
    num_true = float(len(tp_scores) +num_false_negs)
    for i in xrange(num_mix):
        print len(Es_clusters[i]),i,num_mix,roc_fname
        if cs is None:
            scores = (Es_clusters[i] * linear_filters[i]).sum(-1).sum(-1).sum(-1)
        else:
            scores = (Es_clusters[i] * linear_filters[i]).sum(-1).sum(-1).sum(-1) + cs[i]
        tp_scores[tp_scores_idx[i]:
                       tp_scores_idx[i+1]] = scores[labels_clusters[i] >.5]
        fp_scores[fp_scores_idx[i]:
                       fp_scores_idx[i+1]] = scores[labels_clusters[i] <.5]

    tp_scores = np.sort(tp_scores)
    fpr = np.array([
            np.sum(fp_scores >= tp_scores[k])
            for k in xrange(tp_scores.shape[0])])/ num_time_points /.005
    tpr = np.array([
            np.sum(tp_scores >= tp_scores[k])
            for k in xrange(tp_scores.shape[0])])/num_true
    fnr = 1. -tpr
    np.savez('%s%s.npz' % (savedir,roc_fname),fpr=fpr,tpr=tpr)
    if make_plots:
        plot_rocs(tpr,fpr,roc_fname,savedir)
    if verbose:
        print "%s achieves average fpr as: %g" %(roc_fname, np.mean(fpr))
        tpr_fom_set=()
        for i in xrange(1,11):
            tpr_fom_set += (tpr[np.arange(len(fpr))[fpr >= 1./60][-1]],)
        print "%s achieves average tpr as: %g over 1,2,...,10 fps per minute" %(roc_fname, np.mean(tpr_fom_set))

def test_predictors_second_stage_clustered(num_mix,linear_filters,
                                           syllable_string,
                                           thresh_percent,
                                           save_tag,
                                           num_time_points,num_false_negs,savedir,roc_fname,make_plots=False,verbose=False,
                                 cs=None):
    """
    Parameters:
    ===========
    num_mix: int
        Number of mixture components, corresponds to how finely the space is chopped up
    linear_filters: tuple
        Should be a tuple of length num_mix where each entry is a linear predictor which can be used on the data examples (length here of each constituent predictor should correspond to the length of the corresponding cluster)
    Es_clusters: tuple
        Again should be a tuple of length num_mix, these are the clusters
        that we are going to perform detection on using the linear_fitlers
    labels_clusters: tuple
        Another tuple of length num_mix, these tuples contain the labels
    """

    # Es_false_pos_cluster =np.load( '%s%s_false_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))
    # Es_true_pos_cluster =np.load( '%s%s_true_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))

    # need a way to set all of these detector thresholds
    tp_scores_idx = np.hstack(([0],np.cumsum(tuple(np.sum(labels_clusters[i] > .5) for i in xrange(num_mix)))))
    fp_scores_idx = np.hstack(([0],np.cumsum(tuple(np.sum(labels_clusters[i] < .5) for i in xrange(num_mix)))))
    tp_scores = np.empty(tp_scores_idx[-1])
    fp_scores =np.empty(fp_scores_idx[-1])
    num_true = float(len(tp_scores) +num_false_negs)
    for i in xrange(num_mix):
        print len(Es_clusters[i]),i,num_mix,roc_fname
        if cs is None:
            scores = (Es_clusters[i] * linear_filters[i]).sum(-1).sum(-1).sum(-1)
        else:
            scores = (Es_clusters[i] * linear_filters[i]).sum(-1).sum(-1).sum(-1) + cs[i]
        tp_scores[tp_scores_idx[i]:
                       tp_scores_idx[i+1]] = scores[labels_clusters[i] >.5]
        fp_scores[fp_scores_idx[i]:
                       fp_scores_idx[i+1]] = scores[labels_clusters[i] <.5]

    tp_scores = np.sort(tp_scores)
    fpr = np.array([
            np.sum(fp_scores >= tp_scores[k])
            for k in xrange(tp_scores.shape[0])])/ num_time_points /.005
    tpr = np.array([
            np.sum(tp_scores >= tp_scores[k])
            for k in xrange(tp_scores.shape[0])])/num_true
    fnr = 1. -tpr
    np.savez('%s%s.npz' % (savedir,roc_fname),fpr=fpr,tpr=tpr)
    if make_plots:
        plot_rocs(tpr,fpr,roc_fname,savedir)
    if verbose:
        print "%s achieves average fpr as: %g" %(roc_fname, np.mean(fpr))
        tpr_fom_set=()
        for i in xrange(1,11):
            tpr_fom_set += (tpr[np.arange(len(fpr))[fpr >= 1./60][-1]],)
        print "%s achieves average tpr as: %g over 1,2,...,10 fps per minute" %(roc_fname, np.mean(tpr_fom_set))


def plot_rocs(tpr,fpr,plot_name,savedir,use_fnr=False):
    if use_fnr:
        fnr = 1.-tpr
        plt.close('all')
        plt.figure()
        plt.plot(fpr,fnr)
        plt.ylabel('False Negative Rate')
        plt.xlabel('False Positive Rate per Second')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(plot_name)
        plt.savefig('%s%s.png' %(plot_name,savedir))
    else:
        plt.close('all')
        plt.figure()
        plt.plot(fpr,tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate per Second')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title(plot_name)
        plt.savefig('%s%s.png' %(savedir,plot_name))


def get_second_layer_cascade_roc_curves(num_mix,savedir,syllable_string,
                                        thresh_percent,save_tag,
                                        template_tag,
                                        make_plots=False,
                                        verbose=False,
                                        load_data_tag=None):
    print "num_mix=%d" % num_mix
    if load_data_tag is None:
        load_data_tag = save_tag
    cascade_identities = np.load('%s%s_second_layer_cascade_identities_%d_%s.npy' % (savedir,syllable_string,

                                                          num_mix,
                                                               template_tag))
    identity_list = tuple(set(cascade_identities))

    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                               load_data_tag))
    out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,
                                                      thresh_percent,load_data_tag),'rb')
    false_neg_times = pickle.load(out)
    out.close()
    num_false_negs = sum(len(utt) for utt in false_neg_times)
    false_neg_lengths = reduce(lambda x,y: x+y, (tuple(k.true_label_times[1]-k.true_label_times[0] for k in utt) for utt in false_neg_times))
    false_neg_length_counts = np.bincount(false_neg_lengths)
    false_neg_nonzero_lengths = np.nonzero(false_neg_length_counts)[0]

    # for each component
    # we compute what lengths of examples were missed
    # and how many were missed, this allows us to
    # analyze the performance of the detectors from different mixture components
    # as a function of the length of the examples considered
    num_false_negs_components = np.zeros(num_mix)
    false_neg_lengths_components = tuple(
        reduce(lambda x,y: x+y, (
                tuple(k.true_label_times[1]-k.true_label_times[0]
                                       for k in utt if k.max_peak_id==mix_component)
                                 for utt in false_neg_times))
        for mix_component in xrange(num_mix))
    false_neg_component_length_counts = ()

    for mix_component in xrange(num_mix):
        num_false_negs_components[mix_component] = sum( len([k for k in utt if k.max_peak_id == mix_component]) for utt in false_neg_times)
        if len(false_neg_lengths_components[mix_component]) > 0:
            false_neg_component_length_counts += (
                np.bincount(false_neg_lengths_components[mix_component]),)
        else:
            false_neg_component_length_counts += (np.ones(0,dtype=int),)


    num_time_points=float(detection_lengths.sum())


    cascade_fpr_tpr = {}

    for cascade_identity in identity_list:
        false_pos_cascade_scores = np.load('%s%s_false_positive_cascade2_scores_%s.npy' %(
                savedir,
                cascade_identity,
                save_tag)
                                          )
        true_pos_cascade_scores = np.load('%s%s_true_positive_cascade2_scores_%s.npy' %(
                savedir,
                cascade_identity,
                save_tag)
                                      )
        true_pos_cascade_sort_idx = np.argsort(true_pos_cascade_scores)[::-1]
        true_pos_cascade_scores = true_pos_cascade_scores[true_pos_cascade_sort_idx]
        true_pos_cascade_lengths = (np.load('%s%s_true_positive_cascade2_lengths_%s.npy' %(
                savedir,
                cascade_identity,
                save_tag)
                                      )[true_pos_cascade_sort_idx]).astype(int)

        true_pos_length_counts = np.bincount(true_pos_cascade_lengths,minlength=len(false_neg_length_counts))
        if len(true_pos_length_counts) > len(false_neg_length_counts):
            fn_length_counts = np.hstack((false_neg_length_counts,
                                          np.zeros(
                        len(true_pos_length_counts)-len(false_neg_length_counts))))
        else:
            fn_length_counts = false_neg_length_counts.copy()

        true_pos_nonzero_lengths = np.nonzero(true_pos_length_counts)[0]



        num_pos = float(len(true_pos_cascade_scores) + num_false_negs)
        pos_length_counts = fn_length_counts + true_pos_length_counts
        pos_nonzero_lengths = np.nonzero(pos_length_counts)[0]

        if pos_length_counts.sum() != num_pos:
            print "A mistake has been made with the length calculation"
            import pdb; pdb.set_trace()

        if verbose:
            print "num_pos=%g" % num_pos
        true_pos_rates = np.zeros(len(true_pos_cascade_scores))
        true_pos_rates_lengths = np.zeros((len(pos_nonzero_lengths),
                                           len(true_pos_cascade_scores)))
        false_pos_rates = np.zeros(len(true_pos_rates))
        for rank_id, tp_score in enumerate(true_pos_cascade_scores):
            true_pos_rates[rank_id] = np.sum(true_pos_cascade_scores>=tp_score)/num_pos
            # for each length with non-zero counts
            # compute the false positive rate at the given level
            for pos_len_idx, nz_len in enumerate(pos_nonzero_lengths):
                try:
                    true_pos_rates_lengths[pos_len_idx,rank_id] = np.sum(true_pos_cascade_scores[true_pos_cascade_lengths==nz_len]>=tp_score)/float(pos_length_counts[nz_len])
                except: import  pdb; pdb.set_trace()

            false_pos_rates[rank_id] = np.sum(false_pos_cascade_scores >= tp_score)/num_time_points /.005


        np.save('%s%s_tpr_cascade2roc_%s.npy' % (savedir,
                                                   cascade_identity,
                                                      save_tag),
                true_pos_rates)
        np.savez('%s%s_tpr_lengths_cascade2roc_%s.npz' % (savedir,
                                                   cascade_identity,
                                                      save_tag),
                pos_nonzero_lengths=pos_nonzero_lengths,
                pos_length_counts=pos_length_counts,
                true_pos_rates_lengths=true_pos_rates_lengths)

        np.save('%s%s_fpr_cascade2roc_%s.npy' % (savedir,
                                                   cascade_identity,
                                                      save_tag),
                false_pos_rates
                )

        # look at classification rates by length


        # do the cascade computation by components
        for mix_component in xrange(num_mix):
            false_pos_cascade_component_scores = np.load('%s%s_false_positive_cascade2_component_scores_%d_%s.npy' %(
                savedir,
                cascade_identity,mix_component,
                save_tag)
                                          )
            true_pos_cascade_component_scores = np.load('%s%s_true_positive_cascade2_component_scores_%d_%s.npy' %(
                savedir,
                cascade_identity, mix_component,
                save_tag)
                                          )
            true_pos_cascade_component_sort_idx = np.argsort(true_pos_cascade_component_scores)[::-1]
            true_pos_cascade_component_scores = true_pos_cascade_component_scores[true_pos_cascade_component_sort_idx]

            true_pos_cascade_component_lengths = np.load('%s%s_true_positive_cascade2_component_lengths_%d_%s.npy' %(
                savedir,
                cascade_identity,mix_component,
                save_tag)
                                      )[true_pos_cascade_component_sort_idx]

            if len(true_pos_cascade_component_lengths) == 0:
                continue
            if len(false_neg_component_length_counts[mix_component]) > 0:
                true_pos_component_length_counts = np.bincount(true_pos_cascade_component_lengths,minlength=len(false_neg_component_length_counts[mix_component]))
            else:
                try:
                    true_pos_component_length_counts = np.bincount(true_pos_cascade_component_lengths)
                except:
                    import pdb; pdb.set_trace()


            # we want to get the bin count for each length of positive
            # examples these are stored among the true positives and
            # the false negatives, so we want to combine the bin
            # counts which means normalizing for length
            # we use a temporary variable fn_component_length_counts
            # for the false negatives since its usually the one that's too
            # short as the code above corrects for the length of the bin
            # count vector for the true positives
            if len(true_pos_component_length_counts) > len(false_neg_component_length_counts[mix_component]):
                fn_component_length_counts = np.hstack((false_neg_component_length_counts[mix_component],
                                          np.zeros(
                        len(true_pos_component_length_counts)-len(false_neg_component_length_counts[mix_component]))))
            else:
                try:
                    fn_component_length_counts = false_neg_component_length_counts[mix_component].copy()
                except:
                    import pdb; pdb.set_trace()

            true_pos_nonzero_component_lengths = np.nonzero(true_pos_component_length_counts)[0]


            num_pos_component = float(len(true_pos_cascade_component_scores) + num_false_negs_components[mix_component])

            pos_component_length_counts = fn_component_length_counts + true_pos_component_length_counts
            pos_nonzero_component_lengths = np.nonzero(pos_component_length_counts)[0]

            if pos_component_length_counts.sum() != num_pos_component:
                print "A mistake has been made with the length calculation for component %d" % mix_component
                import pdb; pdb.set_trace()


            if verbose:
                print "num_pos_component[%d]=%g" % (mix_component,num_pos_component)

            true_pos_component_rates = np.zeros(len(true_pos_cascade_component_scores))
            true_pos_component_rates_lengths = np.zeros((len(pos_nonzero_component_lengths),
                                               len(true_pos_cascade_component_scores)))

            false_pos_component_rates = np.zeros(len(true_pos_component_rates))
            for rank_id, tp_score in enumerate(true_pos_cascade_component_scores):
                true_pos_component_rates[rank_id] = np.sum(true_pos_cascade_component_scores>=tp_score)/num_pos_component
                for pos_len_idx, nz_len in enumerate(pos_nonzero_component_lengths):
                    true_pos_component_rates_lengths[pos_len_idx,rank_id] = np.sum(true_pos_cascade_component_scores[true_pos_cascade_component_lengths==nz_len]>=tp_score)/float(pos_component_length_counts[nz_len])

                false_pos_component_rates[rank_id] = np.sum(false_pos_cascade_component_scores >= tp_score)/num_time_points /.005

            np.save('%s%s_tpr_component_cascade2roc_%d_%s.npy' % (savedir,
                                                   cascade_identity,mix_component,
                                                      save_tag),
                true_pos_component_rates)
            np.savez('%s%s_tpr_component_lengths_cascade2roc_%d_%s.npz' % (savedir,
                                                   cascade_identity,mix_component,
                                                      save_tag),
                pos_nonzero_lengths=pos_nonzero_component_lengths,
                pos_length_counts=pos_component_length_counts,
                true_pos_rates_lengths=true_pos_component_rates_lengths)

            np.save('%s%s_fpr_component_cascade2roc_%d_%s.npy' % (savedir,
                                                   cascade_identity,mix_component,
                                                      save_tag),
                false_pos_component_rates
                )



        #import pdb; pdb.set_trace()
        if make_plots:
            use_idx = false_pos_rates <= 1.
            plt.close('all')
            plt.plot(false_pos_rates[use_idx],true_pos_rates[use_idx],
                     )
            cascade_fpr_tpr[cascade_identity] =(false_pos_rates[use_idx],
                                                true_pos_rates[use_idx])
            plt.ylabel('Percent True Positives Retained')
            plt.xlabel('False Positives per second')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.title('ROC %s num_mix=%d' %(cascade_identity,
                                                                  num_mix))
            plt.savefig('%s%s_roc_cascade2_%s_%d_%s.png' % (savedir,
                                                            syllable_string,
                                                            cascade_identity,
                                            num_mix,save_tag))
            plt.close('all')

    if make_plots:
        plt.close('all')
        for cascade_identity in identity_list:
            plt.plot(cascade_fpr_tpr[cascade_identity][0],
                     cascade_fpr_tpr[cascade_identity][1],
                     label=cascade_identity)
        plt.legend( tuple(identity_list),prop={'size':6},loc="lower right")
        plt.ylabel('Percent True Positives Retained')
        plt.xlabel('False Positives per second')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('ROC compare %s num_mix=%d' %(cascade_identity,
                                                                  num_mix))
#        plt.show()
        plt.savefig('%sCompare_all_roc_cascade2_%s_%d_%s.png' % (savedir,
                                                            syllable_string,
                                            num_mix,save_tag))
        plt.close('all')
        import re
        SVM_pattern = re.compile('SVM')
        label_list =[]
        for cascade_identity in identity_list:
            if SVM_pattern.search(cascade_identity) is None:
                continue
            plt.plot(cascade_fpr_tpr[cascade_identity][0],
                     cascade_fpr_tpr[cascade_identity][1],
                     label=cascade_identity)
            label_list.append(cascade_identity)
        plt.legend( tuple(label_list),prop={'size':6},loc="lower right")
        plt.ylabel('Percent True Positives Retained')
        plt.xlabel('False Positives per second')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('ROC compare SVM %s num_mix=%d' %(cascade_identity,
                                                                  num_mix))
#        plt.show()
        plt.savefig('%sCompare_SVM_roc_cascade2_%s_%d_%s.png' % (savedir,
                                                            syllable_string,
                                            num_mix,save_tag))
        plt.close('all')
        quantized_pattern = re.compile('quantized')
        label_list =[]
        for cascade_identity in identity_list:
            if quantized_pattern.search(cascade_identity) is None:
                continue
            plt.plot(cascade_fpr_tpr[cascade_identity][0],
                     cascade_fpr_tpr[cascade_identity][1],
                     label=cascade_identity)
            label_list.append(cascade_identity)
        plt.legend( tuple(label_list),prop={'size':6},loc="lower right")
        plt.ylabel('Percent True Positives Retained')
        plt.xlabel('False Positives per second')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('ROC compare quantized %s num_mix=%d' %(cascade_identity,
                                                                  num_mix))
#        plt.show()
        plt.savefig('%sCompare_quantized_roc_cascade2_%s_%d_%s.png' % (savedir,
                                                            syllable_string,
                                            num_mix,save_tag))
        plt.close('all')
        label_list =[]
        for cascade_identity in identity_list:
            if SVM_pattern.search(cascade_identity) is None:
                continue
            plt.plot(cascade_fpr_tpr[cascade_identity][0],
                     cascade_fpr_tpr[cascade_identity][1],
                     label=cascade_identity)
            label_list.append(cascade_identity)

        fpr=np.load('%sfpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    load_data_tag))
        tpr=np.load('%stpr_1stage_%d_%s.npy' % (savedir,num_mix,
                                    load_data_tag))
        label_list.append('no cascade')
        plt.plot(fpr,tpr,label='no cascade')
        plt.legend( tuple(label_list),prop={'size':6},loc="lower right")
        plt.ylabel('Percent True Positives Retained')
        plt.xlabel('False Positives per second')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('ROC compare SVM to no cascade %s num_mix=%d' %(cascade_identity,
                                                                  num_mix))
#        plt.show()
        plt.savefig('%sCompare_SVM_roc_cascade2_against_no_cascade_%s_%d_%s.png' % (savedir,
                                                            syllable_string,
                                            num_mix,save_tag))
        plt.close('all')


def remove_redundant_centers(cluster_centers):
    """
    Clustering process can make clusters overlap, this
    removes redundant ones
    """
    cluster_centers =  np.array(list(set(cluster_centers)))
    return cluster_centers, len(cluster_centers)

def update_affinities_centers(affinities,length_list,cluster_centers):
    """
    Assign each datapoint to a cluster

    Parameters:
    ===========
    affinities: np.ndarray[ndim=2]
        List of which length is assigned to which cluster
    length_list: np.ndarray[ndim=1]
        list of the lengths of the data in order
    cluster_centers: np.ndarray[ndim=1]
    """
    for l_idx, l in enumerate(length_list):
        best_cluster = np.argmin(np.abs(cluster_centers - l))
        affinities[l_idx][:] = 0
        affinities[l_idx,best_cluster] =1

    centers = np.dot(length_list,affinities)/affinities.sum(0)

    return centers



def cluster_lengths(pos_length_counts,pos_nonzero_lengths,num_clusters,
                    verbose=False):
    """
    Simple k-means algorithm to cluster lengths
    Returns the clustered lengths and their indices in pos_nonzero_lengths

    """
    length_list = []
    for nz_len in pos_nonzero_lengths:
        length_list.extend(pos_length_counts[nz_len]*[nz_len])

    length_list = np.array(length_list).astype(float)
    n = len(length_list)
    cluster_centers = length_list[((np.arange(num_clusters)+1.)/(num_clusters+1) *len(length_list)).astype(int)]
    cluster_centers,num_clusters = remove_redundant_centers(cluster_centers)

    affinities = np.zeros((n,num_clusters))
    new_centers = update_affinities_centers(affinities,length_list,cluster_centers)
    new_centers,new_num_clusters = remove_redundant_centers(new_centers)
    while num_clusters != new_num_clusters or np.any(np.sort(new_centers) != np.sort(cluster_centers)):
        num_clusters = new_num_clusters
        cluster_centers = new_centers
        if verbose:
            print cluster_centers
        new_centers = update_affinities_centers(affinities,length_list,cluster_centers)
        new_centers,new_num_clusters = remove_redundant_centers(new_centers)

    length_clusters = ()
    for cur_cluster in xrange(num_clusters):
        length_clusters += (length_list[affinities[:,cur_cluster].astype(bool)],)

    length_sets = [np.array([],dtype=int) for i in xrange(num_clusters)]
    length_indices = [np.array([],dtype=int) for i in xrange(num_clusters)]
    for nz_idx, nz_len in enumerate(pos_nonzero_lengths):
        best_cluster = 0
        best_cluster_count = 0
        for cluster_id in xrange(num_clusters):
            cur_cluster_count = np.sum( length_clusters[cluster_id] == nz_len)
            if cur_cluster_count > best_cluster_count:
                best_cluster = cluster_id
                best_cluster_count = cur_cluster_count
        length_sets[best_cluster] = np.append(length_sets[best_cluster],nz_len)
        length_indices[best_cluster] = np.append(length_indices[best_cluster],nz_idx)


    length_indices_starts = tuple(length_ind[0] for length_ind in length_indices)
    sorted_length_sets_indices = np.argsort(length_indices_starts)
    length_sets = tuple(length_sets[k] for k in sorted_length_sets_indices)
    length_indices = tuple(length_indices[k] for k in sorted_length_sets_indices)


    return length_sets, length_indices, length_list

def plot_component_roc_curves(num_mix,savedir,save_tag,syllable_string,template_tag):
    """
    For each cascade component plot the performance of different mixture components also on lengths
    """
    cascade_identities = np.load('%s%s_second_layer_cascade_identities_%d_%s.npy' % (savedir,syllable_string,

                                                                                     num_mix,
                                                                                     template_tag))
    identity_list = tuple(set(cascade_identities))

    SVM_pattern = re.compile('SVM')
    for cascade_identity in identity_list:

        # only make SVM graphs since the other ones are worthless
        if SVM_pattern.search(cascade_identity) is None:
            continue

        true_pos_rates = np.load('%s%s_tpr_cascade2roc_%s.npy' % (savedir,
                                                   cascade_identity,
                                                      save_tag))
        false_pos_rates = np.load('%s%s_fpr_cascade2roc_%s.npy' % (savedir,
                                                   cascade_identity,
                                                      save_tag))
        outfile = np.load('%s%s_tpr_lengths_cascade2roc_%s.npz' % (savedir,
                                                   cascade_identity,
                                                                   save_tag))
        pos_nonzero_lengths=outfile['pos_nonzero_lengths']
        pos_length_counts=outfile['pos_length_counts']
        true_pos_rates_lengths=outfile['true_pos_rates_lengths']

        length_sets, length_indices, length_list = cluster_lengths(pos_length_counts,pos_nonzero_lengths,10)
        use_bins = np.sort([np.sort(length_set)[0] for length_set in length_sets] + [length_list[-1]])


        base_histogram = np.histogram(length_list,bins=use_bins)
        max_histogram_bin = base_histogram[0].max()

        # want all histograms on the same axis and this guarantees
        # that we get that

        markers = []
        for m in Line2D.markers:
            try:
                if len(m) == 1 and m != '':
                    markers.append(m)
            except TypeError:
                pass

        plt.close('all')

        num_clusters = len(length_sets)
        cur_marker = 1
        for cluster_id in xrange(num_clusters):
                # get the recall rate for true positives
                # that sit within the cluster found by the cluster_lengths function
            if cluster_id % 6 == 0:
                cur_marker += 1
            cluster_tpr = np.dot(pos_length_counts[length_sets[cluster_id]],true_pos_rates_lengths[length_indices[cluster_id]]) / pos_length_counts[length_sets[cluster_id]].sum()
            plt.plot(false_pos_rates,
                     cluster_tpr,marker = markers[cur_marker],
                     label='[%d,%d]' % (length_sets[cluster_id][0],
                                        length_sets[cluster_id][-1]))
        plt.plot(false_pos_rates,
                 true_pos_rates,
                 label='overall')

        plt.legend( prop={'size':6},loc="lower right")
        plt.ylabel('Recall')
        plt.xlabel('False Positives per second')
        plt.xlim([0,.5])
        plt.ylim([0,1])
        plt.title('ROC compare %s num_mix=%d' %(cascade_identity,
                                                num_mix))
#        plt.show()
        plt.savefig('%sCompare_roc_cascade2_%s_lengths_%s_%d_%s.png' % (savedir,cascade_identity,
                                                            syllable_string,
                                            num_mix,save_tag))
        plt.close('all')
        # now we plot a histogram over the lengths

        plt.hist(length_list,bins=use_bins)

        plt.title('Positive example length distribution for %s num_mix=%d' % (cascade_identity,
                                                                                                   num_mix))
        plt.ylim([0,max_histogram_bin])
        plt.savefig('%shistogram_cascade2_%s_lengths_%s_%d_%s.png' % (savedir,cascade_identity,
                                                            syllable_string,
                                            num_mix,save_tag))
        plt.close('all')


        mix_num_pos = ()
        pos_mix_list = []
        for mix_component in xrange(num_mix):
            true_pos_rates = np.load('%s%s_tpr_component_cascade2roc_%d_%s.npy' % (savedir,
                                                   cascade_identity,mix_component,
                                                      save_tag))

            outfile = np.load('%s%s_tpr_component_lengths_cascade2roc_%d_%s.npz' % (savedir,
                                                   cascade_identity,mix_component,
                                                      save_tag))
            pos_nonzero_lengths=outfile['pos_nonzero_lengths']
            pos_length_counts=outfile['pos_length_counts']
            true_pos_rates_lengths=outfile['true_pos_rates_lengths']

            mix_num_pos += (pos_length_counts.sum(),)
            pos_mix_list += mix_num_pos[mix_component] * [mix_component]

            false_pos_rates = np.load('%s%s_fpr_component_cascade2roc_%d_%s.npy' % (savedir,
                                                   cascade_identity,mix_component,
                                                      save_tag))

            # cluster the lengths so that we show the different scores by length
            length_sets, length_indices, length_list = cluster_lengths(pos_length_counts,pos_nonzero_lengths,3)


            plt.close('all')

            num_clusters = len(length_sets)
            for cluster_id in xrange(num_clusters):
                # get the recall rate for true positives
                # that sit within the cluster found by the cluster_lengths function
                cluster_tpr = np.dot(pos_length_counts[length_sets[cluster_id]],true_pos_rates_lengths[length_indices[cluster_id]]) / pos_length_counts[length_sets[cluster_id]].sum()
                plt.plot(false_pos_rates,
                     cluster_tpr,
                     label='[%d,%d]' % (length_sets[cluster_id][0],
                                        length_sets[cluster_id][-1]))
            plt.plot(false_pos_rates,
                     true_pos_rates,
                     label='overall')

            plt.legend( prop={'size':6},loc="lower right")
            plt.ylabel('Recall')
            plt.xlabel('False Positives per second')
            plt.xlim([0,.5])
            plt.ylim([0,1])
            plt.title('ROC compare %s num_mix=%d mix_component=%d' %(cascade_identity,
                                                num_mix,mix_component))
#        plt.show()
            plt.savefig('%sCompare_roc_cascade2_%s_component_lengths_%s_%d_%d_%s.png' % (savedir,cascade_identity,
                                                            syllable_string,
                                            num_mix,mix_component,save_tag))
            plt.close('all')
            # now we plot a histogram over the lengths
            length_sets, length_indices, length_list = cluster_lengths(pos_length_counts,pos_nonzero_lengths,5)

            plt.hist(length_list,bins=use_bins)

            plt.title('Positive example length distribution for %s num_mix=%d mix_component=%d' % (cascade_identity,
                                                                                                   num_mix,mix_component))
            plt.ylim([0,max_histogram_bin])
            plt.savefig('%shistogram_cascade2_%s_component_lengths_%s_%d_%d_%s.png' % (savedir,cascade_identity,
                                                            syllable_string,
                                            num_mix,mix_component,save_tag))
            plt.close('all')

        plt.close('all')
        plt.hist(pos_mix_list,bins=np.arange(num_mix+1))

        plt.savefig('%s%snum_pos_per_component_histogram_%s_%d_%s.png' % (savedir,syllable_string,cascade_identity,num_mix,save_tag))
        plt.close('all')






def get_second_stage_roc_curves_clustered(num_mix,savedir,syllable_string,thresh_percent,
                                save_tag,old_max_detect_tag,make_plots=False,verbose=False,num_binss=np.array([0,3,4,5,7,10,15,23])):
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                save_tag))
    false_neg_scores = np.load('%s%s_false_negative_scores_%d_%d_%s.npy' % (savedir,syllable_string,
                                                             num_mix,
                                                                  thresh_percent,
                                                                  save_tag,
                                                                  ))
    num_false_negs =float(len(false_neg_scores))
    num_time_points=float(detection_lengths.sum())


    outfile= np.load('%slinear_filter_%d_%s.npz'% (savedir,num_mix,old_max_detect_tag))
    base_lfs = tuple(outfile['arr_%d'%i] for i in xrange(num_mix))
    outfile= np.load('%sc_%d_%s.npz'% (savedir,num_mix,old_max_detect_tag))
    base_cs = tuple(outfile['arr_%d'%i] for i in xrange(num_mix))
    like2_lfs =()
    like2_cs =()
    penalty_list=(('unreg', 1),
                                                 ('little_reg',.1),
                                                 ('reg', 0.05),
                                                 ('reg_plus', 0.01),
                                                 ('reg_plus_plus',.001))
    svm_lfs ={}
    svm_bs ={}
    for penalty, val in penalty_list:
        svm_lfs[penalty] =()
        svm_bs[penalty] =()

    quantized_lfs ={}
    quantized_cs ={}
    for num_bins in num_binss:
        quantized_lfs[num_bins] =()
        quantized_cs[num_bins] = ()

    for mix_component in xrange(num_mix):
        outfile = np.load('%s%s_lf_c_second_stage_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                                   num_mix,old_max_detect_tag))
        like2_lfs += (outfile['lf'],)
        like2_cs += (outfile['c'],)
        for penalty,val in penalty_list:
            outfile =np.load('%s%s_w_b_second_stage_%d_%d_%s_%s.npz' % (savedir,syllable_string,
                                                                        mix_component,
                                                                        num_mix,
                                                                        penalty,old_max_detect_tag))
            svm_lfs[penalty] += (outfile['w'].reshape(like2_lfs[-1].shape),)
            svm_bs[penalty] += (outfile['b'],)

        for num_bins in num_binss:
            outfile = np.load('%s%s_lf_c_quantized_second_stage_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,num_bins,
                                                             save_tag))
            quantized_lfs[num_bins] += (outfile['lf'],)
            quantized_cs[num_bins] += (outfile['c'],)


    test_predictors_second_stage_clustered(num_mix,base_lfs,
                                           syllable_string,
                                           thresh_percent,
                                           save_tag,
                                 num_time_points,num_false_negs,savedir,
                                 '%s_base_roc_fpr_tpr_with_cs_%d_%s' % (syllable_string,num_mix,save_tag),
                                 make_plots=make_plots,verbose=verbose,
                                 cs=base_cs)

    test_predictors_second_stage(num_mix,base_lfs,clusters_for_classification,
                                 labels_for_classification,
                                 num_time_points,num_false_negs,savedir,
                                 '%s_base_roc_fpr_tpr_%d_%s' % (syllable_string,num_mix,save_tag),
                                 make_plots=make_plots,verbose=verbose)

    for num_bins in num_binss:
        test_predictors_second_stage(num_mix,quantized_lfs[num_bins],
                                     clusters_for_classification,
                                     labels_for_classification,
                                     num_time_points,num_false_negs,savedir,
                                     '%s_svms_roc_fpr_tpr_%d_%s_%s' % (syllable_string,num_mix,penalty,save_tag),
                                     make_plots=make_plots,verbose=verbose)

        test_predictors_second_stage(num_mix,quantized_lfs[num_bins],
                                     clusters_for_classification,
                                     labels_for_classification,
                                     num_time_points,num_false_negs,savedir,
                                     '%s_svms_roc_fpr_tpr_with_cs_%d_%s_%s' % (syllable_string,num_mix,penalty,save_tag),
                                     make_plots=make_plots,verbose=verbose,
                                     cs=quantized_cs[num_bins])



    for penalty,val in penalty_list:
        print penalty,val
        test_predictors_second_stage(num_mix,svm_lfs[penalty],clusters_for_classification,
                                     labels_for_classification,
                                     num_time_points,num_false_negs,savedir,
                                     '%s_svms_roc_fpr_tpr_%d_%s_%s' % (syllable_string,num_mix,penalty,save_tag),
                                     make_plots=make_plots,verbose=verbose)

        test_predictors_second_stage(num_mix,svm_lfs[penalty],clusters_for_classification,
                                     labels_for_classification,
                                     num_time_points,num_false_negs,savedir,
                                     '%s_svms_roc_fpr_tpr_with_cs_%d_%s_%s' % (syllable_string,num_mix,penalty,save_tag),
                                     make_plots=make_plots,verbose=verbose,
                                     cs=svm_bs[penalty])


    test_predictors_second_stage(num_mix,like2_lfs,clusters_for_classification,
                                 labels_for_classification,
                                 num_time_points,num_false_negs,savedir,
                                 '%s_like2_roc_fpr_tpr_%d_%s' % (syllable_string,num_mix,save_tag),
                                 make_plots=make_plots,verbose=verbose)







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




def get_clustered_examples_by_detect_times(detect_times_fname,
                                           Es_lengths_fname,
                                           Ss_lengths_fname,num_mix,
                                           template_tag,
                                           savedir,
                                           verbose=True,
                                           use_spectral=False):
    """
    Parameters:
    ===========
    detect_times_fname: str
        File containing the detection times, generally going to be a string
        of the form '%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix)
        these are false_pos_times or true_pos_times
    Es_lengths_fname: str
        Path to a file containing the Es and lengths for the detect times
        along the lines of
        '%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
    Ss_lengths_fname: str
        Path to a file containing the Ss and lengths for the detect times
        along the lines of
        '%s%s_false_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
    template_tag: str
        string appended to the end of the template file names we are using
    savedir:
        directory where the templates are saved
    """
    out = open(detect_times_fname,'rb')
    detect_times=pickle.load(out)
    out.close()
    template_ids = rf.recover_template_ids_detect_times(detect_times)
    if verbose:
        print "Es_lengths_fname=%s" % (Es_lengths_fname)
    outfile = np.load(Es_lengths_fname)
    lengths_false_pos = outfile['lengths']
    Es_false_pos = outfile['Es']
    outfile = np.load(Ss_lengths_fname)
    lengths_S_false_pos = outfile['lengths']
    Ss_false_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts

    template_out = get_templates(num_mix,template_tag=template_tag,savedir=savedir, use_spectral=use_spectral)
    if use_spectral:
        templates,sigmas = template_out
    else:
        templates=template_out


    Es_false_pos_clusters = rf.get_false_pos_clusters(Es_false_pos,
                                               templates,
                                               template_ids)
    Ss_false_pos_clusters =rf.get_false_pos_clusters(Ss_false_pos,
                                               templates,
                                               template_ids)
    return Es_false_pos_clusters, Ss_false_pos_clusters



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
    templates = get_templates(num_mix,template_tag=save_tag)
    Es_false_pos_clusters = rf.get_false_pos_clusters(Es_false_pos,
                                               templates,
                                               template_ids)
    Ss_false_pos_clusters =rf.get_false_pos_clusters(Ss_false_pos,
                                               templates,
                                               template_ids)

    if verbose:
        print "num_mix=%d" % num_mix
        print 'Cluster sizes are for false positives %s' % ' '.join(str(k.shape[1]) for k in Es_false_pos_clusters)
    false_pos_cluster_counts = np.array([len(k) for k in Es_false_pos_clusters])
    if verbose:
        for false_pos_idx, false_pos_count in enumerate(false_pos_cluster_counts):
            print "Template %d had %d false positives" %(false_pos_idx,false_pos_count)
    bgd =np.load('%sbgd_%s.npy' %(savedir,save_tag))
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

    detect_times_fname = '%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
    Es_lengths_fname = '%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
    (Es_false_pos_clusters2,
     Ss_false_pos_clusters2) = get_clustered_examples_by_detect_times(detect_times_fname,
                                           Es_lengths_fname,
                                           Ss_lengths_fname,
                                           template_tag,
                                           savedir)

    np.savez('%s%s_Es_false_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Es_false_pos_clusters)
    np.savez('%s%s_Ss_false_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Ss_false_pos_clusters)


    if thresh_percent is None:
        out = open('%s%s_pos_times_%d.pkl' % (savedir,syllable_string,num_mix),'rb')
    else:
        out = open('%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),'rb')
    true_pos_times=pickle.load(out)
    out.close()
    template_ids = rf.recover_template_ids_detect_times(true_pos_times)
    outfile = np.load('%s%s_true_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_true_pos = outfile['lengths']
    Es_true_pos = outfile['Es']
    outfile = np.load('%s%s_true_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag))
    lengths_S_true_pos = outfile['lengths']
    Ss_true_pos = outfile['Ss']
    #
    # Display the spectrograms for each component
    #
    # get the original clustering of the parts
    templates = get_templates(num_mix,template_tag=save_tag)
    Es_true_pos_clusters = rf.get_true_pos_clusters(Es_true_pos,
                                               templates,
                                               template_ids)
    Ss_true_pos_clusters =rf.get_true_pos_clusters(Ss_true_pos,
                                               templates,
                                               template_ids)

    if verbose:
        print "num_mix=%d" % num_mix
        print 'Cluster sizes are for true positives %s' % ' '.join(str(k.shape[1]) for k in Es_true_pos_clusters)
    true_pos_cluster_counts = np.array([len(k) for k in Es_true_pos_clusters])
    if verbose:
        for true_pos_idx, true_pos_count in enumerate(true_pos_cluster_counts):
            print "Template %d had %d true positives" %(true_pos_idx,true_pos_count)
    bgd =np.load('%sbgd_%s.npy' %(savedir,save_tag))


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

    if verbose:
        print "num_mix=%d" % num_mix
        print 'Cluster sizes are for true positives %s' % ' '.join(str(k.shape[1]) for k in clustered_training_true_Es)


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

def train_fp_detector(num_mix,syllable_string,new_tag,thresh_percent=None,save_tag=None,make_plots=False,savedir='data/',template_tag='',
                    verbose=False,cluster_false_pos=False):
    if cluster_false_pos:
        if thresh_percent is None:
            false_pos_times_fname ='%s%s_false_pos_times_%d.pkl' % (savedir,syllable_string,num_mix)
        else:
            false_pos_times_fname ='%s%s_false_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)
        (Es_false_pos_clusters,
         Ss_false_pos_clusters)=get_clustered_examples_by_detect_times(
            false_pos_times_fname,
            '%s%s_false_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),
        '%s%s_false_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),num_mix,
            template_tag,
            savedir)
    else:
        pass

    if verbose:
        print "num_mix=%d" % num_mix
        print 'Cluster sizes are for false positives %s' % ' '.join(str(k.shape[1]) for k in Es_false_pos_clusters)
    false_pos_cluster_counts = np.array([len(k) for k in Es_false_pos_clusters])
    if verbose:
        for false_pos_idx, false_pos_count in enumerate(false_pos_cluster_counts):
            print "Template %d had %d false positives" %(false_pos_idx,false_pos_count)


    np.savez('%s%s_Es_false_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Es_false_pos_clusters)
    np.savez('%s%s_Ss_false_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Ss_false_pos_clusters)


    if thresh_percent is None:
        true_pos_times_fname = '%s%s_pos_times_%d.pkl' % (savedir,syllable_string,num_mix)
    else:
        true_pos_times_fname = '%s%s_pos_times_%d_%d_%s.pkl' % (savedir,syllable_string,num_mix,thresh_percent,save_tag)

    (Es_true_pos_clusters,
     Ss_true_pos_clusters)=get_clustered_examples_by_detect_times(
        true_pos_times_fname,
        '%s%s_true_positives_Es_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),
        '%s%s_true_positives_Ss_lengths_%d_%d_%s.npz' % (savedir,syllable_string,num_mix,thresh_percent,save_tag),num_mix,
        template_tag,
        savedir)

    np.savez('%s%s_Es_true_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Es_true_pos_clusters)
    np.savez('%s%s_Ss_true_pos_clusters_%d_%s.npz' % (savedir,syllable_string,num_mix,new_tag),
             *Ss_true_pos_clusters)
    if verbose:
        print "num_mix=%d" % num_mix
        print 'Cluster sizes are for true positives %s' % ' '.join(str(k.shape[1]) for k in Es_true_pos_clusters)
    true_pos_cluster_counts = np.array([len(k) for k in Es_true_pos_clusters])
    if verbose:
        for true_pos_idx, true_pos_count in enumerate(true_pos_cluster_counts):
            print "Template %d had %d true positives" %(true_pos_idx,true_pos_count)

    templates = get_templates(num_mix,template_tag=template_tag,savedir=savedir)



    for mix_component in xrange(num_mix):
        if verbose:
            print "Working on mixture component: %d" % mix_component
        if make_plots:
            plt.close('all')
            plt.imshow(np.mean(Ss_false_pos_clusters[mix_component],0).T[::-1],
                       origin="lower left")
            plt.savefig('%s%s_Ss_false_pos_template_%d_%d_%s.png' % (savedir,syllable_string,num_mix,mix_component,new_tag))
            plt.close('all')
        get_like_ratio_quantized_second_stage_detection(
            Es_true_pos_clusters[mix_component],
            Es_false_pos_clusters[mix_component],
            templates[mix_component],
            num_mix,
            mix_component,
            syllable_string,save_tag,savedir=savedir,num_binss=np.array([0,3,4,5,7,10,15,23]),
                                        make_plots=make_plots)

        get_baseline_second_stage_detection(Es_true_pos_clusters[mix_component],Es_false_pos_clusters[mix_component],
                                            templates[mix_component], num_mix,mix_component,
                                            syllable_string,new_tag,savedir,
                                            make_plots=make_plots)
        get_svm_second_stage_detection(Es_true_pos_clusters[mix_component],Es_false_pos_clusters[mix_component],
                                            templates[mix_component], num_mix,mix_component,
                                            syllable_string,new_tag,savedir,
                                            make_plots=make_plots)


def train_fp_detector_clustered(num_mix,
                                syllable_string,new_tag,thresh_percent=None,save_tag=None,make_plots=False,savedir='data/',template_tag='',
                    verbose=False,cluster_false_pos=False,num_binss=np.array([0,3,4,5,7,10,15,23]),
                                use_spectral=False,
                                use_percent=1.,
                                use_percent_random_seed=0):

    templates_out = get_templates(num_mix,template_tag=template_tag,savedir=savedir,use_spectral=use_spectral)
    if use_spectral:
        templates, sigmas = templates_out
    else:
        templates = templates_out
    for mix_component in xrange(num_mix):
        if verbose:
            print "Working on mixture component: %d" % mix_component


        try:
            Es_false_pos_cluster =np.load( '%s%s_false_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))
            Ss_false_pos_cluster = np.load('%s%s_false_positives_Ss_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))
            Es_true_pos_cluster =np.load( '%s%s_true_positives_Es_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))
            Ss_true_pos_cluster = np.load('%s%s_true_positives_Ss_lengths_%d_%d_%d_%s.npy' % (savedir,syllable_string,num_mix,mix_component,thresh_percent,save_tag))
        except:
            continue
    #else:
    #    pass

        if use_percent < 1.:
            print "performing subsampling with use_percent=%f" % use_percent

            np.random.seed(use_percent_random_seed)
            fp_subsampled_cluster_ids = np.random.permutation(len(Es_false_pos_cluster))[:
                int(use_percent*len(Es_false_pos_cluster))]
            Es_false_pos_cluster = Es_false_pos_cluster[fp_subsampled_cluster_ids]
            Ss_false_pos_cluster = Ss_false_pos_cluster[fp_subsampled_cluster_ids]
            tp_subsampled_cluster_ids = np.random.permutation(len(Es_true_pos_cluster))[:
                int(use_percent*len(Es_true_pos_cluster))]
            Es_true_pos_cluster = Es_true_pos_cluster[tp_subsampled_cluster_ids]
            Ss_true_pos_cluster = Ss_true_pos_cluster[tp_subsampled_cluster_ids]



        if not use_spectral and Es_false_pos_cluster.shape[1:] != templates[mix_component].shape:
            import pdb; pdb.set_trace()

        if verbose:
            print "num_mix=%d" % num_mix
            print 'Cluster sizes are for false positives %d' % Es_false_pos_cluster.shape[1]
            print 'Temlate-length=%d' % len(templates[mix_component])



        if make_plots:
            plt.close('all')
            plt.imshow(np.mean(Ss_false_pos_cluster,0).T[::-1],
                       origin="lower left")
            plt.savefig('%s%s_Ss_false_pos_template_%d_%d_%s.png' % (savedir,syllable_string,num_mix,mix_component,new_tag))
            plt.close('all')

        if not use_spectral:
            true_responses,false_responses =get_like_ratio_quantized_second_stage_detection(
                Es_true_pos_cluster,
                Es_false_pos_cluster,
                templates[mix_component],
                num_mix,
                mix_component,
                syllable_string,new_tag,savedir=savedir,num_binss=num_binss,
                make_plots=make_plots,
                return_outs=True)

            get_like_ratio_quantized_second_stage_detection(
                Es_true_pos_cluster,
                Es_false_pos_cluster,
                templates[mix_component],
                num_mix,
                mix_component,
                syllable_string,new_tag,savedir=savedir,num_binss=np.array([0,3,4,5,7,10,15,23]),
                                        make_plots=make_plots)


            get_baseline_second_stage_detection(Es_true_pos_cluster,
                                                Es_false_pos_cluster,
                                                templates[mix_component], num_mix,mix_component,
                                                syllable_string,new_tag,savedir,
                                                make_plots=make_plots)
            get_svm_second_stage_detection(Es_true_pos_cluster,
                                           Es_false_pos_cluster,
                                           templates[mix_component], num_mix,mix_component,
                                            syllable_string,new_tag,savedir,
                                            make_plots=make_plots)
        else:
            get_svm_second_stage_detection(Ss_true_pos_cluster,
                                           Ss_false_pos_cluster,
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
    np.save('%s%s_fpr_detector_rocLike_%d_%d_%s.npy' % (savedir,syllable_string,
                                                     mix_component,
                                                     num_mix,save_tag),roc_curve)
    if make_plots:
        plt.close('all')
        plt.plot(1-np.arange(roc_curve.shape[0],dtype=float)/roc_curve.shape[0],roc_curve)
        plt.xlabel('Percent True Positives Retained')
        plt.ylabel('Percent False Positives Retained')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title('ROC %s Likelihood num_mix=%d mix_id=%d' %(syllable_string,
                                                                  num_mix,
                                                                  mix_component))
        plt.savefig('%s%s_fp_roc_discriminationLike_%d_%d_%s.png' % (savedir,syllable_string,
                                                     mix_component,
                                                            num_mix,save_tag))
        plt.close('all')

def quantize_template(template,num_bins,bin_vals=None,bin_lims=None):
    """
    """
    if bin_lims is None:
        bin_lims = np.append(np.arange(num_bins,dtype=float)/num_bins,1.)
    if bin_vals is None:
        bin_vals =(bin_lims[1:] +bin_lims[:-1])/2.

    for bin_id, bin_val in enumerate(bin_vals):
        template[ (template >= bin_lims[bin_id])
                  * (template < bin_lims[bin_id+1])] = bin_val
    return template



def get_like_ratio_quantized_second_stage_detection(true_pos_cluster,false_pos_cluster,
                                        template, num_mix,mix_component,
                                        syllable_string,save_tag,savedir='data/',num_binss=np.array([0,3,4,5,7,10,15,23]),
                                        make_plots=False,verbose=True,
                                                    return_outs=False,):
    """
    Uses a simple bin-based quantization technique, one could imagine also quantizing the false positive template and the true positive
    template by using a v-optimal histogram construction technique

    we do different levels of quantization here to see how it effects the roc curves
    """
    num_false_pos_component =false_pos_cluster.shape[0]
    false_pos_template = np.clip(np.mean(false_pos_cluster[:num_false_pos_component/2],0),.01,.99)
    num_true_pos_component =true_pos_cluster.shape[0]
    true_pos_template= np.clip(np.mean(true_pos_cluster[:num_true_pos_component/2],0),.01,.99)
    for num_bins in num_binss:
        if num_bins < 2:
            fp_quantized = false_pos_template
            tp_quantized = true_pos_template
        else:
            fp_quantized = quantize_template(false_pos_template.copy(),num_bins)
            tp_quantized = quantize_template(true_pos_template.copy(),num_bins)
        if verbose:
            print "num_bins=%d, len(tp_quantized)=%d, len(fp_quantized)=%d" % (num_bins,len(tp_quantized),len(fp_quantized))
        lf,c = et.construct_linear_filter_structured_alternative(
            tp_quantized,
            fp_quantized,

            bgd=None,min_prob=.01)
        np.savez('%s%s_lf_c_quantized_second_stage_%d_%d_%d_%s.npz' % (savedir,syllable_string,
                                                     mix_component,
                                                      num_mix,num_bins,
                                                             save_tag),
                 lf=lf,
                 c=c)
        true_responses = np.sort(np.sum(true_pos_cluster * lf + c,-1).sum(-1).sum(-1))
        false_responses = np.sort((false_pos_cluster[num_false_pos_component/2:]*lf+ c).sum(-1).sum(-1).sum(-1))
        roc_curve = np.array([
                np.sum(false_responses >= true_response)/float(len(false_responses))
                for true_response in true_responses])
        np.save('%s%s_fpr_detector_rocLike_%d_%d_%d_%s.npy' % (savedir,syllable_string,
                                                               mix_component,
                                                               num_mix,num_bins,save_tag),roc_curve)
        if make_plots:
            plt.close('all')
            plt.plot(1-np.arange(roc_curve.shape[0],dtype=float)/roc_curve.shape[0],roc_curve)
            plt.xlabel('Percent True Positives Retained')
            plt.ylabel('Percent False Positives Retained')
            plt.title('ROC %s Likelihood num_mix=%d mix_id=%d num_bins=%d' %(syllable_string,
                                                                  num_mix,
                                                                  mix_component,num_bins))
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.savefig('%s%s_fp_roc_discriminationLike_%d_%d_%d_%s.png' % (savedir,syllable_string,
                                                     mix_component,
                                                            num_mix,num_bins,save_tag))
            plt.close('all')
        if return_outs:
            return true_responses, false_responses


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
        try:
            w = clf.coef_[0]
        except:
            continue
            import pdb; pdb.set_trace()
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
            plt.xlim([0,1])
            plt.ylim([0,1])
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
    print "Finished save_syllable_features_to_data_dir"
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

def get_max_peak_loc_val(detection_row,example_starts_ends,
                         time_bound=np.inf):
    """
    For each tuple in example_start_ends we want to find a peak, or a max value
    and quantify how peaky it is (TODO: not implemented)

    The potential locations for a peak is any time in the segment that the
    detection row corresponds which is closest to the proposed start time
    and whose distance is less than the distance bound

    Parameters:
    ===========
    detection_row: numpy.ndarray[ndim=1]
        detection outputs from a detector, this is the one-dimensional signal in which
        we are hunting peaks
    example_starts_ends: list of tuples of ints
        Each tuple contains a start time, for each start time we are looking
        for a peak

    Returns:
    ========
    peaks: list of (int,float,float)
        First entry of the tuples is the location, the second entry is
        the peak height and the third entry is the 'peakiness'
    """
    peaks =[]
    for example_id,start_end_tuple in enumerate(example_starts_ends):
        start_time, end_time = start_end_tuple
        # get the lower bound
        l_time = int(max(0,start_time-time_bound))
        if example_id >0:
            l_time=int(max(l_time,
                           (start_time + example_starts_ends[example_id-1][0])/2))
        # get the time upperbound
        u_time = int(min(len(detection_row),start_time+time_bound))
        if example_id < len(example_starts_ends) -1:
            u_time=int(min(u_time,
                           (start_time+example_starts_ends[example_id+1][0])/2))
        loc = np.argmax(detection_row[l_time:u_time]) + l_time
        val = detection_row[loc]
        peakiness = get_peakiness(detection_row,loc)
        peaks.append((start_time-loc,val,peakiness))

    return peaks

def get_peakiness(detection_row,loc):
    """
    Get peakiness
    """
    return 1

def analyze_peak_locs_values(num_mix,
                             save_tag,savedir,
                             time_bound=np.inf):
    """
    We load in the detection array and the associated example_start_end_times
    these get passed to a function that, for each positive example,
    we get a tuple where the first entry is the time vector (positive or negative)
    from when the start of the example is and the second entry is peak value
    """
    # load in the detection array, the detection lengths, and the example start and end times
    detection_array = np.load('%sdetection_array_%d_%s.npy' % (savedir,num_mix,
                                                               save_tag))
    detection_lengths = np.load('%sdetection_lengths_%d_%s.npy' % (savedir,num_mix,
                                                                   save_tag))
    out = open('%sexample_start_end_times_%s.pkl' % (savedir,save_tag),'rb')
    example_start_end_times=pickle.load(out)
    out.close()

    all_peaks = []
    for fl_id,example_starts_ends in enumerate(example_start_end_times):
        if len(example_starts_ends) == 0: pass
        all_peaks.extend(get_max_peak_loc_val(detection_array[fl_id,:detection_lengths[fl_id]],example_starts_ends,
                                              time_bound=time_bound))

    return all_peaks

def visualize_peaks(fname,all_peaks,
                    savedir):
    plt.close('all')
    peak_locs, peak_vals, peakinesses =zip(*all_peaks)
    plt.scatter(peak_locs,peak_vals)
    plt.title('%s' % fname)
    plt.xlabel('peak values')
    plt.ylabel('peak_locs')
    plt.savefig('%s%s.png' % (savedir,fname))

def main(args):
    if args.v:
        print args

    syllable_string = '_'.join(p for p in args.detect_object)
    (sp,
     ep,pp,
     root_path,
     test_path,train_path,
     train_example_lengths, train_file_indices,train_classify_lengths,

     test_example_lengths, test_file_indices,test_classify_lengths,

     penalty_list) = get_params(
        sample_rate=args.sample_rate,
        num_window_samples=args.num_window_samples,
        num_window_step_samples=args.num_window_step_samples,
        fft_length=args.fft_length,
        kernel_length=args.kernel_length,
        freq_cutoff=args.freq_cutoff,
        use_mel=args.use_mel,
        do_mfccs=args.do_mfccs,
        no_use_dpss=args.no_use_dpss,
        mel_nbands=args.mel_nbands,
        num_ceps=args.num_ceps,
        liftering=args.liftering,
        include_energy=args.include_energy,
        include_deltas=args.include_deltas,
        include_double_deltas=args.include_double_deltas,
        delta_window=args.delta_window,
        do_freq_smoothing=(not args.no_freq_smoothing),
        block_length=args.block_length,
        spread_length=args.spread_length,
        threshold=args.edge_threshold_quantile,
        magnitude_features=args.magnitude_features,
        use_parts=args.use_parts,
        parts_path=args.parts_path,
        parts_S_path=args.parts_S_path,
        save_parts_S=args.save_parts_S,
        bernsteinEdgeThreshold=args.bernsteinEdgeThreshold,
        spreadRadiusX=args.spreadRadiusX,
        spreadRadiusY = args.spreadRadiusY,
        root_path=args.root_path,
        train_suffix=args.train_suffix,
        test_suffix=args.test_suffix,
        savedir=args.savedir,
        mel_smoothing_kernel=args.mel_smoothing_kernel,
        penalty_list=args.penalty_list,
        partGraph=args.partGraph,
        spreadPartGraph=args.spreadPartGraph)

    print syllable_string
    print "spectrogram parameters:"
    print sp
    if args.svm_name is None:
        svm_name=None
    else:
        svm_name='_'.join(args.svm_name)
    if args.leehon_mapping:
        leehon_mapping, use_phns = get_leehon_mapping()
    else:
        leehon_mapping =None
    if args.save_all_leehon_phones:
        leehon_mapping, rejected_phones, use_phns = get_leehon39_dict()
        jobs = []
        for phn in use_phns:
            if phn in rejected_phones: continue

            p = multiprocessing.Process(target=save_all_leehon_phones(train_path,
                                   train_file_indices,
                                   leehon_mapping, phn,
                                   sp,ep,pp,
                                   args.save_tag,
                                   args.savedir,
                                   args.mel_smoothing_kernel,
                                   args.offset,
                                   10,
                                   num_use_file_idx= args.num_use_file_idx))
            jobs.append(p)
            p.start

    if args.save_syllable_features_to_data_dir:
        save_syllable_features_to_data_dir(args.detect_object,
                                           train_path,
                                           train_file_indices,
                                           sp,ep,
                                           leehon_mapping,
                                           pp=pp,
                                           save_tag=args.save_tag,
                                           waveform_offset=10,
                                           savedir=args.savedir,verbose=args.v,
                                           mel_smoothing_kernel=args.mel_smoothing_kernel,
                                           offset=args.offset,
                                           num_use_file_idx=args.num_use_file_idx)
        print "Finished save_syllable_features_to_data_dir"
    else:
        savedir=args.savedir

    if args.visualize_processed_examples is not None:
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
                                           savedir=args.savedir
                                           )

        visualize_processed_examples(Es,Elengths,Ss,Slengths,syllable_string,
                                     savedir=args.savedir,plot_prefix=args.visualize_processed_examples,pp=pp)
    if args.estimate_templates:
        (Ss,
         Slengths,
         Es,
         Elengths) =get_processed_examples(args.detect_object,
                                           train_path,
                                           train_file_indices,
                                           sp,ep,
                                           leehon_mapping,save_tag=args.save_tag,
                                           waveform_offset=15,
                                           return_waveforms=False,
                                           savedir=args.savedir)
        print "Finished get_processed_examples"
        if len(args.num_mix_parallel) >0:
            estimate_templates(args.num_mix_parallel,
                            Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.template_tag,
                            savedir=args.savedir,
                               do_truncation=(not args.no_template_truncation))
        else:
            estimate_templates((args.num_mix,),
                            Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.template_tag,
                            savedir=args.savedir,
                               do_truncation=(not args.no_template_truncation))

    if args.estimate_templates_limited_data is not None:
        (Ss,
         Slengths,
         Es,
         Elengths) =get_processed_examples(args.detect_object,
                                           train_path,
                                           train_file_indices,
                                           sp,ep,
                                           leehon_mapping,save_tag=args.save_tag,
                                           waveform_offset=15,
                                           return_waveforms=False,
                                           savedir=args.savedir)
        print "Finished get_processed_examples"
        if len(args.num_mix_parallel) >0:
            estimate_templates(args.num_mix_parallel,
                            Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.template_tag,
                            savedir=args.savedir,
                               do_truncation=(not args.no_template_truncation),
                               percent_use=args.estimate_templates_limited_data)
        else:
            estimate_templates((args.num_mix,),
                            Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.template_tag,
                            savedir=args.savedir,
                               do_truncation=(not args.no_template_truncation),
                               percent_use=args.estimate_templates_limited_data)


    if args.estimate_spectral_templates:
        (Ss,
         Slengths,
         Es,
         Elengths) =get_processed_examples(args.detect_object,
                                           train_path,
                                           train_file_indices,
                                           sp,ep,
                                           leehon_mapping,save_tag=args.save_tag,
                                           waveform_offset=15,
                                           return_waveforms=False,
                                           savedir=args.savedir)
        print "Finished get_processed_examples"
        if len(args.num_mix_parallel) >0:
            estimate_spectral_templates(args.num_mix_parallel,
                            Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.template_tag,
                            savedir=args.savedir,
                               do_truncation=(not args.no_template_truncation))
        else:
            estimate_spectral_templates((args.num_mix,),
                                        Es,Elengths,
                            Ss,Slengths,
                            get_plots=args.make_plots,
                            save_tag=args.template_tag,
                            savedir=args.savedir,
                               do_truncation=(not args.no_template_truncation))



    if args.visualize_templates:
        visualize_template(args.num_mix_parallel,syllable_string,
                           args.template_tag,
                           args.savedir,
                           args.save_tag)

    print "Finished estimate_templates"

    if args.get_classification_scores == "train":
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel classification"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p = multiprocessing.Process(target=
                                            get_classification_scores(
                        num_mix,
                        train_classify_lengths,
                        train_path,
                        train_file_indices,
                        sp,ep,pp=pp,save_tag=args.save_tag,
                        template_tag=args.template_tag,
                        savedir=args.savedir,
                        num_use_file_idx=args.num_use_file_idx,
                        use_noise_file=args.use_noise_file,
                        noise_db=args.noise_db,
                        load_data_tag=args.load_data_tag))
                jobs.append(p)
                p.start
    if args.get_classification_scores == "test":
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel classification"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p = multiprocessing.Process(target=
                                            get_classification_scores(
                        num_mix,
                        test_classify_lengths,
                        test_path,
                        test_file_indices,
                        sp,ep,pp=pp,save_tag=args.save_tag,
                        template_tag=args.template_tag,
                        savedir=args.savedir,
                        num_use_file_idx=args.num_use_file_idx,
                        use_noise_file=args.use_noise_file,
                        noise_db=args.noise_db,
                        load_data_tag=args.load_data_tag))
                jobs.append(p)
                p.start

    if args.save_detection_setup == "test":
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p = multiprocessing.Process(target=
                                            save_detection_setup(num_mix,
                                                                 test_example_lengths,
                                                                 test_path,test_file_indices,
                                                                 args.detect_object,sp,
                                                                 ep,
                                                                 leehon_mapping,
                                                                 pp=pp,
                                                                 save_tag=args.save_tag,template_tag=args.template_tag,
                                                                 savedir=args.savedir,verbose=args.v,
                                                                       num_use_file_idx=args.num_use_file_idx,
                                                                       use_svm_based=args.use_svm_based_templates,syllable_string=syllable_string,
                                                                 svm_name=svm_name,
                                                                 use_svm_filter=args.use_svm_filter,
                                                                 use_noise_file=args.use_noise_file,
                                                                 noise_db=args.noise_db,
                                                                 use_spectral=args.do_spectral_detection,
                                                                 load_data_tag=args.load_data_tag))
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
                             ep,leehon_mapping,pp=pp,save_tag=args.save_tag,template_tag=args.template_tag,savedir=args.savedir,verbose=args.v,
                                                                       num_use_file_idx=args.num_use_file_idx,
                                                                       use_svm_based=args.use_svm_based_templates,syllable_string=syllable_string,
                                                                 svm_name=svm_name,
                                                                 use_svm_filter=args.use_svm_filter,
                                                                 use_spectral=args.do_spectral_detection,
                                                                       load_data_tag=args.load_data_tag))
                jobs.append(p)
                p.start
        else:
            save_detection_setup(args.num_mix,train_example_lengths,
                                 train_path,train_file_indices,args.detect_object,sp,
                                 ep,leehon_mapping,pp=pp,save_tag=args.save_tag,template_tag=args.template_tag,savedir=args.savedir,verbose=args.v,
                                                                       num_use_file_idx=args.num_use_file_idx,
                                                                       use_svm_based=args.use_svm_based_templates,syllable_string=syllable_string,
                                                                 svm_name=svm_name,
                                                                 use_svm_filter=args.use_svm_filter,
                                                                 use_spectral=args.do_spectral_detection,
                                 load_data_tag=args.load_data_tag)
    if args.visualize_detection_setup=='train_detect_visualize':
        if len(args.num_mix_parallel) >0 :
            for num_mix in args.num_mix_parallel:
                print num_mix
                visualize_detection_setup(num_mix,train_example_lengths,
                         train_path,train_file_indices[:args.num_use_file_idx],sp,args.visualize_detection_setup,
                        args.savedir,args.save_tag)
        else:
                visualize_detection_setup(args.num_mix,train_example_lengths,
                         train_path,train_file_indices,sp,args.visualize_detection_setup,
                        args.savedir,args.save_tag)

    elif args.visualize_detection_setup=='test_detect_visualize':
        if len(args.num_mix_parallel) >0 :
            for num_mix in args.num_mix_parallel:
                print num_mix
                visualize_detection_setup(num_mix,test_example_lengths,
                         test_path,test_file_indices[:args.num_use_file_idx],sp,args.visualize_detection_setup,
                        args.savedir,args.save_tag)
        else:
                visualize_detection_setup(args.num_mix,test_example_lengths,
                         test_path,train_file_indices,sp,args.visualize_detection_setup,
                        args.savedir,args.save_tag)
    if args.get_fpr_tpr_classify == 'train':
        leehon_mapping, rejected_phones, use_phns = get_leehon39_dict(no_sil=args.no_sil)
        jobs = []
        for num_mix in args.num_mix_parallel:
            p=multiprocessing.Process(target=
            get_fpr_tpr_classify(num_mix,train_classify_lengths,
                                 args.detect_object,
                                 leehon_mapping,
                                 rejected_phones,
                                 use_phns,
                                 data_type='train',
                                 save_tag=args.save_tag,
                                 savedir=args.savedir,
                                 get_plots=args.make_plots,
                                 use_spectral=args.do_spectral_detection,
                                 template_tag=args.template_tag)
                                 )
            jobs.append(p)
            p.start
    if args.get_max_classification_results == 'train':
        leehon_mapping, rejected_phones, use_phns = get_leehon39_dict(no_sil=args.no_sil)
        jobs = []
        for num_mix in args.num_mix_parallel:
            p = multiprocessing.Process(target=
                                        get_max_classification_results(
                    num_mix,
                    args.save_tag_suffix,
                    "train",args.savedir,
                    use_phns,rejected_phones,leehon_mapping,
                    train_classify_lengths,verbose=args.v))
            jobs.append(p)
            p.start

    if args.get_classify_confusion_matrix == 'train':
        leehon_mapping, rejected_phones, use_phns = get_leehon39_dict(no_sil=args.no_sil)
        jobs = []
        for num_mix in args.num_mix_parallel:
            p = multiprocessing.Process(target=
                                        get_classify_confusion_matrix(
                    num_mix,
                    args.save_tag,args.savedir,
                    "train",
                    use_phns,
                    verbose=args.v))
            jobs.append(p)
            p.start

    if args.get_classify_scores_metadata == 'train':
        leehon_mapping, rejected_phones, use_phns = get_leehon39_dict(no_sil=args.no_sil)
        jobs = []
        for num_mix in args.num_mix_parallel:
            p = multiprocessing.Process(target=
                                        get_classify_scores_metadata(
                    num_mix,args.detect_object[0],
                    args.save_tag,args.savedir,
                    "train",
                    use_phns, train_classify_lengths,
                    verbose=args.v))
            jobs.append(p)
            p.start

    if args.get_fpr_tpr_tagged:
        print "Finished save_detection_setup"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p=multiprocessing.Process(target=get_fpr_tpr_tagged(num_mix,syllable_string,
                           return_detected_examples=False,
                           return_clusters=False,
                           save_tag=args.save_tag,savedir=args.savedir,
                           get_plots=args.make_plots,
                                                                    old_max_detect_tag=args.old_max_detect_tag,
                               use_spectral=args.do_spectral_detection,
                               template_tag=args.template_tag))
                jobs.append(p)
                p.start
        else:
            get_fpr_tpr_tagged(args.num_mix,syllable_string,
                               return_detected_examples=False,
                               return_clusters=False,
                               save_tag=args.save_tag,savedir=args.savedir,
                               get_plots=True,
                                                                    old_max_detect_tag=args.old_max_detect_tag,
                               use_spectral=args.do_spectral_detection,
                               template_tag=args.template_tag)

    if args.get_tagged_all_detection_clusters:
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
                                           get_tagged_all_detection_clusters(num_mix,args.save_tag,args.old_max_detect_tag,savedir=args.savedir))
                jobs.append(p)
                p.start
        else:
            get_tagged_all_detection_clusters(args.num_mix,args.save_tag,args.old_max_detect_tag,savedir=args.savedir)

    if args.get_tagged_detection_clusters:
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
                                           get_tagged_detection_clusters(num_mix,args.thresh_percent,save_tag=args.save_tag,use_thresh=None,old_max_detect_tag=args.old_max_detect_tag,savedir=args.savedir,
                                                                         use_spectral=args.do_spectral_detection,
                                                                         template_tag=args.template_tag))
                jobs.append(p)
                p.start
        else:
            get_tagged_detection_clusters(args.num_mix,args.thresh_percent,save_tag=args.save_tag,use_thresh=None,old_max_detect_tag=args.old_max_detect_tag,savedir=args.savedir,
                                          use_spectral=args.do_spectral_detection,
                                          template_tag=args.template_tag)

    if args.plot_detection_outs != '':
        print "Plotting the detection outputs"
        plot_detection_outs(args.plot_detection_outs,args.num_mix,sp,ep,test_path,test_file_indices,
                            save_tag=args.save_tag,template_tag='train_2',
                            savedir=args.savedir,
                            verbose=args.v)

    if args.get_detection_clusters_by_label == "train":
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:

                out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (args.savedir,num_mix,
                                                                                args.thresh_percent,
                                                                                args.save_tag)
                                                                    ,'rb')
                detection_clusters = cPickle.load(out)
                out.close()
                (pos_times,
                 false_pos_times,
                 false_neg_times,
                 example_types,
                 ) = get_detection_clusters_by_label(num_mix,train_path,
                                                     train_file_indices,args.thresh_percent,single_threshold=True,save_tag=args.save_tag,verbose=args.v,
                                                     savedir=args.savedir,
                                                     return_example_types=True, old_max_detect_tag=args.old_max_detect_tag,template_tag=args.template_tag,
                                                     detect_clusters=detection_clusters,
                                                     use_spectral=args.do_spectral_detection)
                print '%s%s_false_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                              args.thresh_percent,args.save_tag)
                templates_out=get_templates(num_mix,template_tag=args.template_tag,savedir=args.savedir,use_spectral=args.do_spectral_detection)
                if args.do_spectral_detection:
                    templates,sigmas = templates_out
                else:
                    templates = templates_out
                template_lengths = np.array([len(t) for t in templates])
                all_lengths ={}
                for utt_id, utt in enumerate(false_pos_times):
                    for fp_id, fp in enumerate(utt):
                        for k_id, k in enumerate(fp.cluster_detect_ids):
                            all_lengths[k]=fp.cluster_detect_lengths[k_id]

                print num_mix, len(template_lengths)

                for k in xrange(num_mix):
                    try:
                        print 'Template_length[%d]=%d, false_pos_template_length[%d]=%d' %(k,template_lengths[k],k,all_lengths[k])
                    except:
                        print 'Template_length[%d]=%d, false_pos_template_length[%d]=' %(k,template_lengths[k],k)
                out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                              args.thresh_percent,args.save_tag),'wb')
                pickle.dump(false_pos_times,out)
                out.close()
                out = open('%s%s_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                            args.thresh_percent,args.save_tag),'wb')
                pickle.dump(pos_times,out)
                out.close()
                out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                                  args.thresh_percent,args.save_tag),'wb')
                pickle.dump(false_neg_times,out)
                out.close()

        else:
            out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (args.savedir,args.num_mix,
                                                                       args.thresh_percent,
                                                                       args.save_tag)
                                                                    ,'rb')
            detection_clusters = cPickle.load(out)
            out.close()
            (pos_times,
             false_pos_times,
             false_neg_times,
             example_types,
             ) = get_detection_clusters_by_label(args.num_mix,train_path,
                                             train_file_indices,args.thresh_percent,single_threshold=True,save_tag=args.save_tag,verbose=args.v,
                                                     savedir=args.savedir,return_example_types=True, old_max_detect_tag=args.old_max_detect_tag,template_tag=args.template_tag,
                                         detect_clusters=detection_clusters,
                                                     use_spectral=args.do_spectral_detection)
            out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,args.num_mix,
                                                              args.thresh_percent,args.save_tag),'wb')
            pickle.dump(false_pos_times,out)
            out.close()
            out = open('%s%s_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,args.num_mix,
                                                args.thresh_percent,args.save_tag),'wb')
            pickle.dump(pos_times,out)
            out.close()
            out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,args.num_mix,
                                                      args.thresh_percent,args.save_tag),'wb')
            pickle.dump(false_neg_times,out)
            out.close()
    if args.get_detection_clusters_by_label == "test":
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:

                out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (args.savedir,num_mix,
                                                                                args.thresh_percent,
                                                                                args.save_tag)
                                                                    ,'rb')
                detection_clusters = cPickle.load(out)
                out.close()

                (pos_times,
                 false_pos_times,
                 false_neg_times,
                 example_types,
                 ) = get_detection_clusters_by_label(num_mix,test_path,
                                                     test_file_indices,args.thresh_percent,
                                                     single_threshold=True,save_tag=args.save_tag,
                                                     verbose=args.v,
                                                     savedir=args.savedir,
                                                     return_example_types=True, old_max_detect_tag=args.old_max_detect_tag,template_tag=args.template_tag,
                                                     detect_clusters=detection_clusters,
                                                     use_spectral=args.do_spectral_detection)
                print '%s%s_false_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                              args.thresh_percent,args.save_tag)
                templates_out=get_templates(num_mix,template_tag=args.template_tag,savedir=args.savedir,use_spectral=args.do_spectral_detection)
                if args.do_spectral_detection:
                    templates,sigmas = templates_out
                else:
                    templates = templates_out
                template_lengths = np.array([len(t) for t in templates])
                all_lengths ={}
                for utt_id, utt in enumerate(false_pos_times):
                    for fp_id, fp in enumerate(utt):
                        for k_id, k in enumerate(fp.cluster_detect_ids):
                            all_lengths[k]=fp.cluster_detect_lengths[k_id]
                print num_mix, len(template_lengths)
                for k in xrange(num_mix):
                    try:
                        print 'Template_length[%d]=%d, false_pos_template_length[%d]=%d' %(k,template_lengths[k],k,all_lengths[k])
                    except: pass

                out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                              args.thresh_percent,args.save_tag),'wb')
                pickle.dump(false_pos_times,out)
                out.close()
                out = open('%s%s_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                            args.thresh_percent,args.save_tag),'wb')
                pickle.dump(pos_times,out)
                out.close()
                out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,num_mix,
                                                                  args.thresh_percent,args.save_tag),'wb')
                pickle.dump(false_neg_times,out)
                out.close()

        else:
            out = open('%sdetection_clusters_single_thresh_%d_%d_%s.pkl' % (args.savedir,args.num_mix,
                                                                       args.thresh_percent,
                                                                       args.save_tag)
                                                                    ,'rb')
            detection_clusters = cPickle.load(out)
            out.close()
            (pos_times,
             false_pos_times,
             false_neg_times,
             example_types,
             ) = get_detection_clusters_by_label(args.num_mix,test_path,
                                             test_file_indices,args.thresh_percent,single_threshold=True,save_tag=args.save_tag,verbose=args.v,
                                                 savedir=args.savedir,
                                                 return_example_types=True, old_max_detect_tag=args.old_max_detect_tag,template_tag=args.template_tag,
                                         detect_clusters=detection_clusters,
                                                     use_spectral=args.do_spectral_detection)
            out = open('%s%s_false_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,args.num_mix,
                                                              args.thresh_percent,args.save_tag),'wb')
            pickle.dump(false_pos_times,out)
            out.close()
            out = open('%s%s_pos_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,args.num_mix,
                                                args.thresh_percent,args.save_tag),'wb')
            pickle.dump(pos_times,out)
            out.close()
            out = open('%s%s_false_neg_times_%d_%d_%s.pkl' % (args.savedir,syllable_string,args.num_mix,
                                                      args.thresh_percent,args.save_tag),'wb')
            pickle.dump(false_neg_times,out)
            out.close()

    if args.get_false_pos_examples:
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
        get_false_pos_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,
                               thresh_percent=args.thresh_percent,save_tag=args.save_tag,
                               savedir=args.savedir,
                               verbose=args.v,
                              pp=pp,
                               num_use_file_idx=args.num_use_file_idx))
                jobs.append(p)
                p.start

    if args.cluster_false_pos_examples:
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
        cluster_false_pos_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,
                               thresh_percent=args.thresh_percent,save_tag=args.save_tag,template_tag=args.template_tag,
                               savedir=args.savedir,
                               verbose=args.v,
                              pp=pp,
                               num_use_file_idx=args.num_use_file_idx,
                                   max_num_points_cluster=args.max_num_points_cluster,
                                   use_spectral=args.do_spectral_detection))
                jobs.append(p)
                p.start

    if args.cluster_true_pos_examples:
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
        cluster_true_pos_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,
                               thresh_percent=args.thresh_percent,save_tag=args.save_tag,template_tag=args.template_tag,
                               savedir=args.savedir,
                               verbose=args.v,
                              pp=pp,
                               num_use_file_idx=args.num_use_file_idx,
                                   max_num_points_cluster=args.max_num_points_cluster,
                                   use_spectral=args.do_spectral_detection))
                jobs.append(p)
                p.start

    if args.get_false_neg_examples:
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
        get_false_neg_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,
                               thresh_percent=args.thresh_percent,save_tag=args.save_tag,
                               savedir=args.savedir,
                               verbose=args.v,
                              pp=pp))
                jobs.append(p)
                p.start

    if args.get_true_pos_examples:
        print args.num_mix
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=
        get_true_pos_examples(num_mix,syllable_string,
                               sp,ep,waveform_offset=10,
                               thresh_percent=args.thresh_percent,save_tag=args.save_tag,
                               savedir=args.savedir,
                               verbose=args.v,
                              pp=pp))
                jobs.append(p)
                p.start

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
    if args.run_fp_detector:
        print "running fp detector"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=run_fp_detector(num_mix,syllable_string,args.save_tag,make_plots=args.make_plots,
                                                                  thresh_percent=args.thresh_percent,save_tag=args.save_tag,savedir=args.savedir,verbose=args.v))
                jobs.append(p)
                p.start

    if args.train_fp_detector:
        print "running fp detector"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=train_fp_detector(num_mix,syllable_string,args.save_tag,make_plots=args.make_plots,
                                                                  thresh_percent=args.thresh_percent,save_tag=args.save_tag,savedir=args.savedir,
                                                                    template_tag=args.template_tag,verbose=args.v))
                jobs.append(p)
                p.start

    if args.train_fp_detector_clustered:
        print "running fp detector"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []

            if args.new_tag is not None:
                new_tag = args.new_tag
            else:
                new_tag = args.save_tag
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=train_fp_detector_clustered(num_mix,syllable_string,new_tag,make_plots=args.make_plots,
                                                                  thresh_percent=args.thresh_percent,save_tag=args.save_tag,savedir=args.savedir,
                                                                    template_tag=args.template_tag,verbose=args.v,
                                                                              use_spectral=args.do_spectral_detection,
                                                                              use_percent=args.train_fp_detector_use_percent,
                                                                              use_percent_random_seed=0))
                jobs.append(p)
                p.start

    if args.save_second_layer_cascade:
        print "saving second layer cascade"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=save_second_layer_cascade(num_mix,syllable_string,
                              args.save_tag,args.template_tag,args.savedir,
                              num_binss=args.num_binss,
                              penalty_list=penalty_list,
                              verbose=args.v,
                                                                            only_svm=args.only_svm,
                                                                            use_spectral=args.do_spectral_detection,
                                                                            old_max_detect_tag=args.old_max_detect_tag))
                jobs.append(p)
                p.start

    if args.false_pos_examples_cascade_score:
        print "saving second layer cascade"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=false_pos_examples_cascade_score(num_mix,syllable_string,
                                     sp,ep,waveform_offset=10,thresh_percent=args.thresh_percent,save_tag=args.save_tag,template_tag=args.template_tag,savedir=args.savedir,
                               verbose=args.v,pp=pp,num_use_file_idx=args.num_use_file_idx,
                               max_num_points_cluster=args.max_num_points_cluster,
                                                                                   do_spectral_detection=args.do_spectral_detection,
                                                                                   num_extract_top_false_positives=args.num_extract_top_false_positives,
                                                                                   old_max_detect_tag=args.old_max_detect_tag,
                                                                                   load_data_tag=args.load_data_tag))
                jobs.append(p)
                p.start

    if args.true_pos_examples_cascade_score:
        print "saving second layer cascade"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=true_pos_examples_cascade_score(num_mix,syllable_string,
                                     sp,ep,waveform_offset=10,thresh_percent=args.thresh_percent,save_tag=args.save_tag,template_tag=args.template_tag,savedir=args.savedir,
                               verbose=args.v,pp=pp,num_use_file_idx=args.num_use_file_idx,
                               max_num_points_cluster=args.max_num_points_cluster,
                                                                                  do_spectral_detection=args.do_spectral_detection,
                                                                                  old_max_detect_tag=args.old_max_detect_tag,
                                                                                  load_data_tag=args.load_data_tag))
                jobs.append(p)
                p.start

    if args.get_second_layer_cascade_roc_curves:
        print "getting cascade second layer roc curves"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=get_second_layer_cascade_roc_curves(num_mix,args.savedir,syllable_string,
                                        args.thresh_percent,args.save_tag,
                                                                                      args.template_tag,
                                        make_plots=args.make_plots,
                                        verbose=args.v,
                                                                                      load_data_tag=args.load_data_tag))
                jobs.append(p)
                p.start


    if args.perform_test_phase_detection:
        print "performing test phase detection"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=test_phase_detection(num_mix,test_example_lengths,test_path,
                                 test_file_indices,
                                 args.detect_object,sp,ep,leehon_mapping,
                                 args.save_tag,
                                 args.savedir,
                                 args.old_max_detect_tag,
                                 args.thresh_percent,
                                 verbose=args.v))
                jobs.append(p)
                p.start
    if args.get_test_output:
        print "Getting test_outputs"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs = []
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=get_test_outputs(num_mix,syllable_string,sp,ep,
                     args.thresh_percent,args.savedir,args.save_tag))
                jobs.append(p)
                p.start
    if args.get_final_test_rocs:
        print "Getting final test rocs"
        if len(args.num_mix_parallel) > 0:
            print "doing parallel"
            jobs =[]
            for num_mix in args.num_mix_parallel:
                p =multiprocessing.Process(target=get_final_test_rocs(num_mix,syllable_string,args.thresh_percent,args.savedir,args.save_tag,args.old_max_detect_tag,make_plots=args.make_plots,verbose=args.v))
                jobs.append(p)
                p.start

    if args.visualize_peaks is not None:
        all_peaks = analyze_peak_locs_values(args.num_mix,
                                             args.save_tag,
                                             args.savedir,
                                             args.peak_bound)
        visualize_peaks(args.visualize_peaks,
                        all_peaks,
                        args.savedir)

    if args.plot_component_roc_curves:
        if len(args.num_mix_parallel)> 0:
            for num_mix in args.num_mix_parallel:
                plot_component_roc_curves(num_mix,args.savedir,args.save_tag,
                                          syllable_string,args.template_tag)
        else:
            plot_component_roc_curves(args.num_mix,args.savedir,args.save_tag,
                                          syllable_string,args.template_tag)


def get_final_test_rocs(num_mix,syllable_string,thresh_percent,savedir,save_tag,old_max_detect_tag,make_plots=False,verbose=False):
    perform_second_stage_detection_testing(num_mix,syllable_string,save_tag,thresh_percent,savedir,
                                           make_plots=False,verbose=False,
                                           old_max_detect_tag=old_max_detect_tag)
    get_second_stage_roc_curves(num_mix,savedir,syllable_string,thresh_percent,
                                save_tag,old_max_detect_tag,make_plots=make_plots,verbose=verbose)


def get_test_outputs(num_mix,syllable_string,sp,ep,
                     thresh_percent,savedir,save_tag,pp=None):
    get_false_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag=save_tag,
                           savedir=savedir,
                           verbose=True,pp=pp)
    get_true_pos_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag=save_tag,
                          savedir=savedir,
                           verbose=True,pp=pp)
    get_false_neg_examples(num_mix,syllable_string,
                           sp,ep,waveform_offset=10,
                           thresh_percent=thresh_percent,save_tag=save_tag,
                           savedir=savedir,
                           verbose=True,pp=pp)


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
                                    test_file_indices,thresh_percent,single_threshold=True,save_tag='test',
                                         verbose=False,
                                         savedir=args.savedir,
                                         return_example_types=True,
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
    templates =get_templates(num_mix,template_tag=template_tag,savedir=savedir)
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
            ax2.plot(
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
    parser.add_argument('--mel_smoothing_kernel',default=-1,type=int,
                        metavar='N',help="The smoothing kernel length, default is -1 so no smoothing is done")
    parser.add_argument('--do_spectral_detection',action='store_true',
                        help="include if you want to do spectral detection")
    parser.add_argument('--do_mfccs',action='store_true',
                        help="Include if you want to use MFCC features")
    parser.add_argument('--mel_nbands',type=int,default=40,
                        help="Number of mel filters to use, default is 40")
    parser.add_argument('--num_ceps',type=int,default=13,
                        help="Number of cepstral coefficients to use for smoothing or for the MFCCs")
    parser.add_argument('--liftering',type=float,default=.6,
                        help="Liftering factor for smoothing the cepstrum")
    parser.add_argument('--include_energy',action='store_true',
                        help='Include if you want energy included')
    parser.add_argument('--include_deltas',action='store_true',
                        help="Include if you want delta features computed on the spectrogram")
    parser.add_argument('--include_double_deltas',action='store_true',
                        help="Include if you want double delta features computed on the spectrogram")
    parser.add_argument('--delta_window',type=int,default=9,
                        help="window over which the delta features are computed in the MFCCs")
    parser.add_argument('--no_use_dpss',action='store_true',
                        help="whether to use the dpss in the computation of the spectrogram")
    parser.add_argument('--no_freq_smoothing',action='store_true',
                        help="whether to do the frequency smoothing, of particular interest in the case of using the multitaper spectral analysis since much smoothing is already included in that")
    parser.add_argument('--block_length',nargs=1,default=40,
                        type=int,metavar='N',
                        help="Blocks that we compute the adaptive edge threshold over")
    parser.add_argument('--spread_length',nargs=1,default=1,
                        type=int,metavar='N',
                        help="Amount of spreading to do for the edge features")
    parser.add_argument('--edge_threshold_quantile',nargs=1,default=.7,
                        type=float,metavar='X',
                        help="Quantile to threshold the edges at, defaults to .7")
    parser.add_argument('--use_parts',action='store_true',
                        help="include flag if you want to use parts for the feature processing")
    parser.add_argument('--parts_path',metavar='Path',
                        type=str,help="Path to the file where the parts are saved or will be saved",
                        default="/home/mark/Template-Speech-Recognition/"
                        + "Development/102012/"
                        + "E_templates.npy")
    parser.add_argument('--parts_S_path',metavar='Path',
                        type=str,help="Path to the file where the parts are saved or will be saved",
                        default="/home/mark/Template-Speech-Recognition/"
                        + "Development/102012/"
                        + "S_clusters.npy")
    parser.add_argument('--save_parts_S',action='store_true',
                        help='whether to save parts')
    parser.add_argument('--partGraph',metavar='Path',
                        type=str,help="Path to file where the part graph is saved",
                        default='/home/mark/Template-Speech-Recognition/'
                        + 'Notebook/25/data/kl_part_graph_30.npy')
    parser.add_argument('--spreadPartGraph',action='store_true',
                        help="whether to spread using the part graph")
    parser.add_argument('--bernsteinEdgeThreshold',metavar='N',
                        type=int,default=12,
                        help="Threshold for the edge counts before a part is fit to a location")
    parser.add_argument('--spreadRadiusX',metavar='N',
                        type=int,default=2,
                        help="An integer. The radius along the time axis which we spread part detections")
    parser.add_argument('--spreadRadiusY',metavar='N',
                        type=int,default=2,
                        help="An integer. The radius along the frequency axis which we spread part detections")
    parser.add_argument('--root_path',metavar='Path',
                        type=str,default='/home/mark/Template-Speech-Recognition/',help='A string that is the path to where the experiments and data are located')
    parser.add_argument('--train_suffix',metavar='Path',type=str,default='Data/Train/',help='A string that when appended to the root path string is the path to where the training data is located')
    parser.add_argument('--test_suffix',metavar='Path',type=str,default='Data/Test/',help='A string that when appended to the root_path is the path to where the testing data is located')
    parser.add_argument('--save_syllable_features_to_data_dir',
                        action='store_true',
                        help="If included this will attempt to save training data estimated using the parameters to the included path for later processing in an experiment.  This includes spectrograms, edgemaps, and waveforms. Defaults to ")
    parser.add_argument('--save_all_leehon_phones',
                        action='store_true',
                        help="Get the features for the lee-hon phones")
    parser.add_argument('--make_plots',action='store_true',
                        help="Ubiquitous argument for whether or not to make plots in whatever functions are called. Defaults to False.")
    parser.add_argument('-v',action='store_true',
                        help="Ubiquitous command that says whether to make the program run in verbose mode with lots of printouts, defaults to False")
    parser.add_argument('--leehon_mapping',action='store_true',
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
    parser.add_argument('--load_data_tag',
                        type=str,metavar='str',default='train',
                        help="Tag for distinguishing which data to open for things like the cascade processing")
    parser.add_argument('--new_tag',default=None,type=str,
                        help="Tag to save things to, this is particularly important for the train_fp_detector_clustered where this will determine the saved name of the cascade detectors")
    parser.add_argument('--train_fp_detector_use_percent',default=1.,type=float,
                        help="percentage of data to use in estimating the false positive detector, this option is here to discover the sample complexity of learning")
    parser.add_argument('--save_detection_setup',default='',
                        type=str,metavar='str',
                        help="Says whether to store the detection array and run save_detection_setup: values are either 'train' or 'test', this affects which set of the data is used")
    parser.add_argument('--get_classification_scores',default='',
                        type=str,metavar='str',
                        help="Gets the classification scores for the train or the test set depending on whether the keyword argument is `train` or `test`")
    parser.add_argument('--estimate_templates',action='store_true',
                        help="Whether to run estimate_templates and save those templates or not")
    parser.add_argument('--estimate_templates_limited_data',type=float,default=None,
                        help="default is none and no limits are placed on the data, when an argument is included this means that a certain fraction of the data won't be used, the template is saved using template_tag, be sure to use that option so that you can use the limited data template in the future")
    parser.add_argument('--estimate_spectral_templates',action='store_true',
                        help="if included runs estimate_spectral_templates which means that templates are estimated using the EM algorithm but we use Gaussian Mixture-based clustering of the spectrogram/mfcc data.  Hence this is done in a continuous domain rather than the discrete domain implied by estimate_templates.")
    parser.add_argument('--plot_detection_outs',default="",type=str,
                        help="whether to run the plot_detection_outs runs with no arguments")
    parser.add_argument('--get_fpr_tpr_tagged',action="store_true",
                        help="whether get_fpr_tpr_tagged to run the plot_detection_outs runs with no arguments"
                        )
    parser.add_argument('--get_fpr_tpr_classify',default='',type=str,
                        help="whether get_fpr_tpr_classify to run the plot_detection_outs runs with no arguments, the argument should be train or test"
                        )
    parser.add_argument('--get_max_classification_results',default='',
                        type=str,
                        help="whether to run the max_classification_results function and the string should be 'train' or 'test' to indicate which data set to use")
    parser.add_argument('--get_classify_confusion_matrix',default='',
                        type=str,
                        help="whether to run the get_classify_confusion_matrix function and the string should be 'train' or 'test' to indicate which data set to use")
    parser.add_argument('--get_classify_scores_metadata',default='',
                        type=str,
                        help="whether to run the get_classify_scores_metadata function and the string should be 'train' or 'test' to indicate which data set to use.  This function also takes in a phn and it gives the set of false positives, true positives, and false negatives")

    parser.add_argument('--save_tag_suffix',type=str,default='train_edges',
                        help="used for get_max_classification_results as the suffix for the save tag where the prefix is the phn of interest.  Default is 'train_edges'")
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
    parser.add_argument('--max_fpr_sec',default=-1,type=int,metavar='N',
                        help="in collecting false positives this willcorrespond to the maximal allowable false positive rate")
    parser.add_argument('--log_file',default='main_multiprocessing.log',
                        type=str,metavar='Path',help="which logfile to use to look at the logging outputs")
    parser.add_argument('--waveform_offset',default=15,type=int,
                        metavar='N',help="Number of frames to pad the waveform vectors by in order to get samples of the sounds")
    parser.add_argument('--train_second_stage_detectors',action="store_true",
                        help="Include this flag if you want to call train_second_stage_detectors")
    parser.add_argument('--get_test_output',action="store_true",
                        help="Include this flag if you want to get all instances of the true/false positives/negatives from the baseline detector over the test data")
    parser.add_argument('--perform_test_phase_detection',action="store_true",
                        help="Include this flag if you want to do the test phase all at once")
    parser.add_argument('--get_final_test_rocs',action="store_true",
                        help="Include this flag if you want to get the final roc curves once all the data has been processed")
    parser.add_argument('--plot_component_roc_curves',
                        action='store_true',
                        help="creates many plots that show how the different components of the classifier perform on the dataset and on the recall rate of examples of different lengths")
    parser.add_argument('--get_detection_clusters_by_label',type=str,default='',
                        help="Include this flag to just run this particular function for diagnosing the clustered detections takes one variable which is either train or test")
    parser.add_argument('--get_false_pos_examples',action="store_true",
                        help="Include this flag to just run this particular function for retrieving false positive examples")
    parser.add_argument('--get_true_pos_examples',action="store_true",
                        help="Include this flag to just run this particular function for retrieving true positive examples")
    parser.add_argument('--get_false_neg_examples',action="store_true",
                        help="Include this flag to just run this particular function for retrieving false negative examples")
    parser.add_argument('--run_fp_detector',action="store_true",
                        help="Gets the basic SVM and second stage models setup for detection")
    parser.add_argument('--train_fp_detector',action="store_true",
                        help="Gets the basic SVM and second stage models setup for detection, users a better setup than run_fp_detector")
    parser.add_argument('--train_fp_detector_clustered',action="store_true",
                        help="Gets the basic SVM and second stage models setup for detection, users a better setup than run_fp_detector, assumes that the true and false positives have already been clustered into components")
    parser.add_argument('--get_tagged_detection_clusters',action="store_true",
                        help="Gets the basic detection clusters for recognition")
    parser.add_argument('--no_template_truncation',action='store_true',
                        help='when estimating the templates for the examples removes the truncation step for the templates')
    parser.add_argument('--visualize_processed_examples',type=str,default=None,
                        help='include this flag and a string to take all the stored examples and create plots of them with the string as a prefix for the plot name')
    parser.add_argument('--visualize_templates',action='store_true',
                        help='include this flag and a string to take all the stored templates and create plots of them with the string as a prefix for the plot name')
    parser.add_argument('--num_use_file_idx',default=-1,type=int,
                        help='An integer, specifies the number of utterances to use in save_detection_setup so that way errors can be identified quickly, default is -1, which means that all files are used')
    parser.add_argument('--visualize_detection_setup',type=str,default='',
                        help='A string, empty means nothing is done, if the string is non-empty saves a file using the string as a prefix for the name where the file is a picture of the detector output and the spectrogram')
    parser.add_argument('--get_tagged_all_detection_clusters',
                        action='store_true',
                        help='no arguments, if flag is included this retrieves all of the detection clusters at every possible threshold and saves them.')
    parser.add_argument('--template_tag',default='train_parts',type=str,
                        help='tag to put on the end of templates for loading them')
    parser.add_argument('--visualize_peaks',type=str,
                        default=None,help="Default is none, otherwise its a string, which will be in the file name for a file that is generated that relates peaks to start times of detect objects")
    parser.add_argument('--peak_bound',type=np.float,
                        default=np.inf,help="Bound for when to count a peak is associated with the start time of an acoustic object")
    parser.add_argument('--offset',default=0,
                        type=int,
                        help="size of offset for where the example lines up")
    parser.add_argument('--cluster_false_pos_examples',action='store_true',
                        help="does a clustering of the false positive examples for me to work with")
    parser.add_argument('--cluster_true_pos_examples',action='store_true',
                        help="does a clustering of the true positive examples for me to work with")
    parser.add_argument("--max_num_points_cluster",default=2000,type=int,
                        help="Maximum number of points in true_pos, false_pos, and false_neg clusters")
    parser.add_argument('--penalty_list',default=['unreg', '1.0',
                                                 'little_reg','0.1',
                                                 'reg', '0.05',
                                                 'reg_plus', '0.01',
                                                 'reg_plus_plus','0.001'],
                        nargs='*',help="List of penalties and names for training the SVM")
    parser.add_argument('--save_second_layer_cascade',action='store_true',
                        help="whether to save the second layer cascade for further use")
    parser.add_argument('--num_binss',type=int,nargs='*',default=[0,3,4,5,7,10,15,23],help='Number of bins for quantization')
    parser.add_argument('--false_pos_examples_cascade_score',action='store_true',help='runs the second layer cascade over the extracted false positives')
    parser.add_argument('--num_extract_top_false_positives',type=int,
                        default=0,
                        help="number of the top scoring false positives to extract and make wave files from")
    parser.add_argument('--true_pos_examples_cascade_score',action='store_true',help='runs the second layer cascade over the extracted true positives')
    parser.add_argument('--get_second_layer_cascade_roc_curves',action='store_true',help='gets the final rocs for the cascaded detectors')
    parser.add_argument('--use_svm_based_templates',action='store_true',
                        help='indicates to use the templates and background inferred from the SVM for doing detection')
    parser.add_argument('--svm_name',nargs='*',default=None,
                        help="name of the SVM used for estimating templates")
    parser.add_argument('--use_svm_filter',nargs='*',default=None,
                        help="two parts of the file path to the svm estimated template")
    parser.add_argument('--use_noise_file',default=None,type=str,
                        help="path to a noise file if you want it added to the data during the test phase of save detection setup")
    parser.add_argument('--noise_db',default=0,type=float,
                        help="compute a multiplier for the noise file so that way we get noise at the right decibel level")
    parser.add_argument('--magnitude_features',action='store_true',
                        help="whether to use magnitude features rather than the edgemap features for base consideration")
    parser.add_argument('--magnitude_and_edge_features',action='store_true',
                        help="whether to use both magnitude and edge features together")
    parser.add_argument('--magnitude_block_length',default=0,type=int,
                        help="block length to use in estimation of the blocks for getting the magnitude feature quantization")
    parser.add_argument('--only_svm',action='store_true',
                        help='only use svm for second stage')
    parser.add_argument('--no_sil',action='store_true',
                        help='add silence to the rejected phone list')
    syllable=('aa','r')
    threshval = 100
    make_plots =True
    print parser.parse_args()
    main(parser.parse_args())
