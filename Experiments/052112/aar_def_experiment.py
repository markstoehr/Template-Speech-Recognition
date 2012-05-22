root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl
import template_speech_rec.parts_model as pm


edge_feature_row_breaks = np.load(root_path+'Experiments/050812/edge_feature_row_breaks.npy')
edge_orientations = np.load(root_path+'Experiments/050812/edge_orientations.npy')
abst_threshold = np.load(root_path+'Experiments/050812/abst_threshold.npy')

texp = template_exp.\
    Experiment(pattern=np.array(('aa','r')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )
output = open('aar_train_tune_data_iter052112.pkl','rb')
train_data_iter = cPickle.load(output)
tune_data_iter = cPickle.load(output)
output.close()


mean_background = np.load(root_path+'Experiments/050812/mean_background_liy051012.npy')




aar_template = np.load(root_path+'Experiments/052112/aar_template_052112.npy')

# construct the aar_template classifier
# for the coarse factor we put it at 1 just because we want to keep comparable to alexey
aar_tpm = pm.TwoPartModel(aar_template,
                          mean_background,
                          part_length = 2*aar_template.shape[1]/3,)
                          
classifier = cl.Classifier(aar_tpm,coarse_factor=1,bg = mean_background)
classifier_flat = cl.Classifier(aar_template,coarse_factor=1,bg = mean_background)

np.save('aar_template_052112',template)
np.save('registered_examples_aar_052112',registered_examples)

data_iter = tune_data_iter

texp = template_exp.\
    Experiment(pattern=np.array(('aa','r')),
               data_paths_file=root_path+'Data/WavFilesTestPaths_feverfew',
               spread_length=3,
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )
test_data_iter, test_tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=1.)

data_iter = test_data_iter
flat_threshold = 1000.
if True:
    allowed_overlap = .3
    edge_feature_row_breaks= np.array([   0.,   
                                          45.,   
                                          90.,  
                                          138.,  
                                          186.,  
                                          231.,  
                                          276.,  
                                          321.,  
                                          366.])
    edge_orientations=np.array([[ 1.,  0.],
                                [-1.,  0.],
                                [ 0.,  1.],
                                [ 0., -1.],
                                [ 1.,  1.],
                                [-1., -1.],
                                [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                              .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positives = []
    all_negatives = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                if classifier_flat.score_no_bg(E_segment[:,:classifier_flat.window[1]]) > flat_threshold:
                    scores[d] = classifier.score_no_bg(E_segment)
            # now we get the indices sorted
            indices = cl.remove_overlapping_examples(np.argsort(scores)[::-1],
                                                     classifier.window[1],
                                                     int(allowed_overlap*classifier.window[1]))
            positives, negatives =  cl.get_pos_neg_scores(indices,pattern_times,
                                                          scores,classifier.window[1])
            all_positives.extend(positives)
            all_negatives.extend(negatives)
        else:
            break

aar_def_roc,aar_def_roc_vals = cl.get_roc(np.sort(all_positives)[::-1],
                                  np.sort(all_negatives)[::-1],num_frames)

np.save('aar_def_roc_vals052112',aar_def_roc_vals)

