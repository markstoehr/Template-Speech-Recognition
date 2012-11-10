import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import matplotlib.pyplot as plt
import cPickle,os, pickle,collections,itertools

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


phn='aa'

perform_phn_template_estimation(phn,utterances_path,
                                    file_indices,sp,ep,
                                    num_mix_params,
                                    phn_mapping=leehon_mapping,
                                    waveform_offset=15)


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
            np.save('data/%d_templates_%s.npy' % (num_mix,phn),
                    templates)
            np.save('data/%d_spec_templates_%s.npy' % (num_mix,phn),
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
            np.savez('data/%d_templates_%s.npz' % (num_mix,phn),
                    *templates)
            np.savez('data/%d_spec_templates_%s.npz' % (num_mix,phn),
                    *spec_templates)
            



def perform_phn_train_detection_SVM(phn, num_mix_params,
                                    train_example_lengths,bgd,
                                    train_path):
    FOMS = collections.defaultdict(list)
    for num_mix in num_mix_params:
        outfile = np.load('%d_templates.npz' % num_mix)
        templates = tuple( outfile['arr_%d'%i] for i in xrange(len(outfile.files)))
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

def extract_false_positives(num_mix_params,first_pass_fns):
    for num_mix in num_mix_params:
        affinities = np.load('data/%d_affinities.npy' % (num_mix))
        for fnr in first_pass_fnrs:
            out = open('data/false_pos_times_%d_%d.pkl' % (num_mix,fnr),'rb')
            false_pos_times=pickle.load(out)
            out.close()
            false_positives = rf.get_false_positives(false_pos_times,
                                                     S_config=sp,
                                                     E_config=ep,
                                                     offset=0,
                                                     waveform_offset=7)
            example_mat = gtrd.recover_example_map(false_positives)
            false_pos_assigned_phns,false_pos_phn_contexts,utt_paths,file_idx,start_ends = gtrd.recover_assigned_phns(false_positives,example_mat)
            np.savez('data/false_pos_phns_assigned_contexts_%d_%d.npy' % (num_mix,fnr),
                    example_mat=example_mat,
                    assigned_phns=false_pos_assigned_phns,
                    phn_contexts=false_pos_phn_contexts,
                     utt_paths=utt_paths,
                     file_idx=file_idx,
                     start_ends=start_ends)
            np.save('data/false_positives_example_mat_%d_%d.npy' % (num_mix,fnr),example_mat)
            lengths,waveforms  = gtrd.recover_waveforms(false_positives,example_mat)
            np.savez('data/false_positives_waveforms_lengths_%d_%d.npz' % (num_mix,fnr),
                     lengths=lengths,
                     waveforms=waveforms)
            Slengths,Ss  = gtrd.recover_specs(false_positives,example_mat)
            np.savez('data/false_positives_Ss_lengths_%d_%d.npz' % (num_mix,fnr),
                     lengths=Slengths,
                     Ss=Ss)
            Elengths,Es  = gtrd.recover_edgemaps(false_positives,example_mat)
            np.savez('data/false_positives_Es_lengths_%d_%d.npz' % (num_mix,fnr),
                     lengths=Elengths,
                     Es=Es)
            templates = et.recover_different_length_templates(affinities,
                                                      Es,
                                                      Elengths)
            spec_templates = et.recover_different_length_templates(affinities,
                                                                   Ss,
                                                                   Slengths)
            cluster_counts = np.zeros(num_mix)
            for mix_id in xrange(num_mix):
                os.system("mkdir -p data/%d_%d_%d" % (num_mix,fnr,mix_id))
            for example_id in xrange(waveforms.shape[0]):
                if affinities[example_id].max() > .999:
                    cluster_id = np.argmax(affinities[example_id])
                    wavfile.write('data/%d_%d_%d/%d.wav' % (num_mix,fnr,cluster_id,cluster_counts[cluster_id]),16000,((2**15-1)*waveforms[example_id]).astype(np.int16))
                    cluster_counts[cluster_id] += 1


# load in the data set
num_mix_params= [2, 3, 4, 5, 6, 7, 8, 9]

def get_SVM_LR_filters(phn,num_mix_params,
                       first_pass_fnrs,
                       assignment_threshold=.95):
    for num_mix in num_mix_params:
        for fnr in first_pass_fnrs:
            # want to know which mixture component we're comparing against
            out = open('data/false_pos_times_%d_%d.pkl' % (num_mix,fnr),'rb')
            false_pos_times=pickle.load(out)
            out.close()
            template_ids = rf.recover_template_ids_detect_times(false_pos_times)
            outfile = np.load('data/false_positives_Es_lengths_%d_%d.npz' % (num_mix,fnr))
            lengths_false_pos = outfile['lengths']
            Es_false_pos = outfile['Es']
            outfile = np.load('data/false_positives_Ss_lengths_%d_%d.npz' % (num_mix,fnr))
            lengths_S_false_pos = outfile['lengths']
            Ss_false_pos = outfile['Ss']
            outfile = np.load('data/false_pos_phns_assigned_contexts_%d_%d.npy' % (num_mix,fnr))
            false_pos_assigned_phns = outfile['assigned_phns']
            false_pos_phn_contexts = outfile['phn_contexts']
            utt_paths = outfile['utt_paths']
            file_idx = outfile['file_idx']
            start_ends=outfile['start_ends']
            #
            # Display the spectrograms for each component
            #
            # get the original clustering of the parts
            outfile = np.load('data/Es_lengths.npz')
            lengths_true_pos = outfile['Elengths']
            example_mat = outfile['example_mat']
            Es_true_pos = outfile['Es']
            outfile = np.load('%d_templates.npz' % num_mix)
            E_templates = tuple(outfile['arr_%d' %i ] for i in xrange(len(outfile.files)))
            false_pos_clusters = rf.get_false_pos_clusters(Es_false_pos,
                                                           E_templates,
                                                           template_ids)
            template_affinities = np.load('%d_affinities.npy' % num_mix)
            clustered_training = et.recover_clustered_data(template_affinities,
                                                           Es_true_pos,
                                                           E_templates,
                                                           assignment_threshold = assignment_threshold)
            # np.savez('data/training_true_pos_%d.npz' %num_mix, *clustered_training)
            # np.savez('data/training_false_pos_%d.npz' % num_mix,*false_pos_clusters)
            # learn a template on half the false positive data
            # do for each mixture component and see the curve
            # need to compute the likelihood ratio test
            for mix_component in xrange(num_mix):
                num_false_pos_component =false_pos_clusters[mix_component].shape[0]
                false_pos_template = np.clip(np.mean(false_pos_clusters[mix_component][:num_false_pos_component/2],0),.05,.95)
                np.save('data/false_pos_template_%d_%d_%d.npy' % (num_mix,
                                                                  mix_component,
                                                                  fnr),
                        false_pos_template)
                lf,c = et.construct_linear_filter_structured_alternative(
                    E_templates[mix_component],
                    false_pos_template,
                bgd=None,min_prob=.01)
                np.save('data/fp_LR_lf_%d_%d_%d' %(num_mix,mix_component,fnr),
                        lf)
                np.save('data/fp_LR_c_%d_%d_%d' %(num_mix,mix_component,fnr),
                        c)
                true_responses = np.sort(np.sum(clustered_training[mix_component] * lf + c,-1).sum(-1).sum(-1))
                false_responses = np.sort((false_pos_clusters[mix_component][num_false_pos_component/2:]*lf+ c).sum(-1).sum(-1).sum(-1))
                open('data/fp_records_%d_%d_%d_%s.dat' %(num_mix,mix_component,fnr,phn),'w').write(
                    '\n'.join(('Assigned Phn\tUtterance Path\tFile Idx\tStart time\tEnd Time\tScore',) 
                              + tuple('%s\t%s\t%s\t%d\t%d\t%g' %
                                      (assigned_phn,utt_path,fl_id,se[0],se[1],f_response)
                                      for assigned_phn,utt_path,fl_id,se,f_response in itertools.izip(false_pos_assigned_phns[template_ids=mix_component],
                                                                                                      utt_paths[template_ids=mix_component],
                                                                                                      file_idx[template_ids=mix_component],
                                                                                                      start_ends[template_ids=mix_component],
                                                                                                      false_responses))))
                roc_curve = np.array([
                    np.sum(false_responses >= true_response)
                    for true_response in true_responses]) 
                np.save('data/fp_detector_roc_%d_%d_%d.npy' % (num_mix,
                                                           mix_component,
                                                           fnr),roc_curve)



SVMResult = collections.namedtuple('SVMResult',
                                   ('num_mix'
                                    +' mix_component'
                                    +' C'
                                    +' W'
                                    +' b'
                                    +' roc_curve'
                                    +' total_error_rate'))


def SVM_training
                                    
def return_svm_result_tuple(num_mix_params,first_pass_fnrs):
    svmresult_tuple= ()
    for num_mix in num_mix_params:
        for fnr in first_pass_fnrs:
            outfile = np.load('data/training_true_pos_%d.npz' % num_mix)
            true_pos_clusters = tuple(outfile['arr_%d'%i] for i in xrange(num_mix))
            del outfile
            outfile = np.load('data/training_false_pos_%d.npz' % num_mix)
            false_pos_clusters = tuple(outfile['arr_%d'%i] for i in xrange(num_mix))
            del outfile
        for mix_component in xrange(num_mix):
        data_shape = true_pos_clusters[mix_component][0].shape
        num_true = len(true_pos_clusters[mix_component])
        num_false = len(false_pos_clusters[mix_component])
        num_true_train = int(num_true * .75)
        num_false_train = int(num_false * .5)
        training_data_X = np.hstack((
                np.ones((num_true_train+num_false_train,1)),
                np.vstack((
                        true_pos_clusters[mix_component][:num_true_train].reshape(
                            num_true_train,
                            np.prod(data_shape)),
                        false_pos_clusters[mix_component][:num_false_train].reshape(
                            num_false_train,
                            np.prod(data_shape))))))
        training_data_Y = np.hstack((
                np.ones(num_true_train),
                np.zeros(num_false_train)))
        testing_data_X = np.hstack((
                np.ones((num_true-num_true_train
                         + num_false - num_false_train,
                         1)),
                np.vstack((
                        true_pos_clusters[mix_component][num_true_train:].reshape(
                            num_true - num_true_train,
                            np.prod(data_shape)),
                        false_pos_clusters[mix_component][num_false_train:].reshape(
                            num_false - num_false_train,
                            np.prod(data_shape))))))
        testing_data_Y = np.hstack((
                np.ones(num_true-num_true_train),
                np.zeros(num_false-num_false_train)))
        for name, penalty in (('unreg', 1), ('little_reg',.1), ('reg', 0.05),('reg_plus', 0.01),
                              ('reg_plus_plus',.001)):
            clf = svm.SVC(kernel='linear', C=penalty)
            clf.fit(training_data_X, training_data_Y)
            # get the roc curve
            w = clf.coef_[0]
            testing_raw_outs = (testing_data_X * w).sum(1)
            val_thresholds = np.sort(testing_raw_outs[testing_data_Y==1])
            roc_curve = np.zeros(len(val_thresholds))
            num_neg = float(np.sum(testing_data_Y==0))
            for i,thresh in enumerate(val_thresholds):
                roc_curve[i] = np.sum(testing_raw_outs[testing_data_Y==0]  <thresh)/num_neg
            plt.figure()
            plt.clf()
            plt.plot(1-np.arange(roc_curve.shape[0])/float(roc_curve.shape[0]),1-roc_curve)
            plt.xlabel('Percent True Positives Retained')
            plt.ylabel('Percent False Positives Retained')
            plt.title('ROC AAR penalty=%g num_mix=%d mix_id=%d' %(penalty,
                                                                  num_mix,
                                                                  mix_component))
            plt.savefig('roc_layer_2_%s_%d_%d' %(name,
                                                                  num_mix,
                                                                  mix_component))
            # get the separating hyperplane
            test_predict_Y = clf.predict(testing_data_X)
            svmresult_tuple += (SVMResult(
                    num_mix=num_mix,
                    mix_component = mix_component,
                    C=penalty,
                    W=w[1:].reshape(data_shape),
                    b=w[0],
                    roc_curve=roc_curve,
                    total_error_rate=np.sum(test_predict_Y == testing_data_Y)/float(len(testing_data_Y))),)

