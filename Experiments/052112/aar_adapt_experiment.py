#
# purpose of this experiment is to see if the adaptive background over the window does better than the basic
#
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl


output = open('aar_train_tune_data_iter052112.pkl','rb')
train_data_iter = cPickle.load(output)
tune_data_iter = cPickle.load(output)
output.close()


mean_background = np.load(root_path+'Experiments/050812/mean_background_liy051012.npy')

# construct the aar_template classifier
# for the coarse factor we put it at 1 just because we want to keep comparable to alexey
classifier = cl.Classifier(template,coarse_factor=1,bg = mean_background)

np.save('aar_template_052112',template)
np.save('registered_examples_aar_052112',registered_examples)

data_iter = tune_data_iter
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
    all_positives_adapt_bg = []
    all_negatives_adapt_bg = []
    all_positives_mel = []
    all_negatives_mel = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                          max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores_adapt_bg = -np.inf * np.ones(num_detections)
            scores_mel = -np.inf * np.ones(num_detections)
            bg = mean_background.copy()
            E_mel = esp.get_edgemap_no_threshold(train_data_iter.s,
                            train_data_iter.sample_rate,
                        train_data_iter.num_window_samples,
                        train_data_iter.num_window_step_samples,
                        train_data_iter.fft_length,8000,7,
                                                 use_mel = True)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                E_segment_mel = E_mel[:,d:d+classifier.window[1]].copy()
                esp.threshold_edgemap(E_segment_mel,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment_mel,edge_feature_row_breaks,edge_orientations,spread_length=3)
                bg = np.minimum(.4,
                                 np.maximum(np.mean(E_segment,axis=1),
                                            .1))
                scores_adapt_bg[d] = classifier.score(E_segment,bg)
                scores_mel[d] = classifier.score_no_bg(E_segment_mel)
            # now we get the indices sorted
            indices_adapt_bg = cl.remove_overlapping_examples(np.argsort(scores_adapt_bg)[::-1],
                                                              classifier.window[1],
                                                              int(allowed_overlap*classifier.window[1]))
            indices_mel = cl.remove_overlapping_examples(np.argsort(scores_mel)[::-1],
                                                            classifier.window[1],
                                                            int(allowed_overlap*classifier.window[1]))
            positives_adapt_bg, negatives_adapt_bg =  cl.get_pos_neg_scores(indices_adapt_bg,
                                                                            pattern_times,
                                                                            scores_adapt_bg,
                                                                            classifier.window[1])
            positives_mel, negatives_coarse =  cl.get_pos_neg_scores(indices_mel,
                                                                        pattern_times,
                                                                        scores_mel,
                                                                        classifier.window[1])
            all_positives_adapt_bg.extend(positives_adapt_bg)
            all_negatives_adapt_bg.extend(negatives_adapt_bg)
            all_positives_mel.extend(positives_mel)
            all_negatives_mel.extend(negatives_mel)
        else:
            break

aar_roc_adapt_bg,aar_roc_vals_adapt_bg = cl.get_roc(np.sort(all_positives_adapt_bg)[::-1],
                                                    np.sort(all_negatives_adapt_bg)[::-1],num_frames)
aar_roc_coarse,aar_roc_vals_coarse = cl.get_roc(np.sort(all_positives_coarse)[::-1],
                                                np.sort(all_negatives_coarse)[::-1],num_frames)

np.save('aar_roc_vals_adapt_bg052112',aar_roc_vals_adapt_bg)
np.save('aar_roc_vals_coarse052112',aar_roc_vals_coarse)

# threshold is going to be 1000
pos = np.sort(all_positives)[::-1]
neg = np.sort(all_negatives)[::-1]
    num_frames = float(num_frames)
    roc_vals = np.zeros(len(pos))
    cur_neg_idx = 0
    while pos[0] <= neg[cur_neg_idx]: cur_neg_idx += 1
    roc_vals[0] = cur_neg_idx/num_frames
    end_loop = False
    for roc_idx in xrange(1,roc_vals.shape[0]):
        if pos[roc_idx] < neg[-1]:
            end_loop = True
        else:
            while pos[roc_idx] <= neg[cur_neg_idx]:
                cur_neg_idx +=1
                if cur_neg_idx >= neg.shape[0]:
                    end_loop = True
                    break
        if end_loop:
            for roc_idx_prime in xrange(roc_idx,roc_vals.shape[0]):
                roc_vals[roc_idx_prime] = -np.inf
            break
        else:
            roc_vals[roc_idx] = cur_neg_idx/num_frames        






    liy_roc,liy_roc_vals = get_roc(np.sort(all_positives)[::-1],
                        np.sort(all_negatives)[::-1],
                        num_frames)

