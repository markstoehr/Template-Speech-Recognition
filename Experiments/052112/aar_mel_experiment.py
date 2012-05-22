root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl


edge_feature_row_breaks = np.load(root_path+'Experiments/050812/edge_feature_row_breaks.npy')
edge_orientations = np.load(root_path+'Experiments/050812/edge_orientations.npy')
abst_threshold = np.load(root_path+'Experiments/050812/abst_threshold.npy')


output = open('aar_train_tune_data_iter052112.pkl','rb')
train_data_iter = cPickle.load(output)
tune_data_iter = cPickle.load(output)
output.close()


train_data_iter.use_mel = True

aar_patterns = []
train_data_iter.reset_exp()
for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_patterns=True,
                            max_template_length=40):
        # the context length is 11
        for p in train_data_iter.patterns:
            pattern = p.copy()
            esp.threshold_edgemap(pattern,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
            esp.spread_edgemap(pattern,edge_feature_row_breaks,edge_orientations,spread_length=5)
            aar_patterns.append(pattern)
    else:
        break

_,_ ,\
        registered_examples,template \
        = et.simple_estimate_template(aar_patterns)

def get_training_template(train_data_iter):
    patterns = []
    train_data_iter.reset_exp()
    for datum_id in xrange(train_data_iter.num_data):
        if datum_id % 10 == 0:
            print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_patterns=True,
                            max_template_length=40):
        # the context length is 11
        for p in train_data_iter.patterns:
            pattern = p.copy()
            esp.threshold_edgemap(pattern,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
            esp.spread_edgemap(pattern,edge_feature_row_breaks,edge_orientations,spread_length=5)
            patterns.append(pattern)
    else:
        break
    _,_ ,\
        registered_examples,template \
        = et.simple_estimate_template(patterns)
    return registered_examples, template



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
    all_positives = []
    all_negatives = []
    all_positives_adapt_bg = []
    all_negatives_adapt_bg = []
    all_positives_coarse = []
    all_negatives_coarse = []
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
            scores_adapt_bg = -np.inf * np.ones(num_detections)
            coarse_scores = -np.inf * np.ones(num_detections)
            bg = mean_background.copy()
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                
                scores[d] = classifier.score_no_bg(E_segment)
                scores_adapt_bg[d] = classifier.score(E_segment,bg)
                coarse_scores[d] = classifier.coarse_score_count(E_segment)
                bg = np.minimum(.1,
                                 np.maximum(np.mean(E_segment,axis=1),
                                            .4))
            # now we get the indices sorted
            indices = cl.remove_overlapping_examples(np.argsort(scores)[::-1],
                                                     classifier.window[1],
                                                     int(allowed_overlap*classifier.window[1]))
            indices_adapt_bg = cl.remove_overlapping_examples(np.argsort(scores_adapt_bg)[::-1],
                                                              classifier.window[1],
                                                              int(allowed_overlap*classifier.window[1]))
            indices_coarse = cl.remove_overlapping_examples(np.argsort(coarse_scores)[::-1],
                                                            classifier.window[1],
                                                            int(allowed_overlap*classifier.window[1]))
            positives, negatives =  cl.get_pos_neg_scores(indices,pattern_times,
                                                          scores,classifier.window[1])
            positives_adapt_bg, negatives_adapt_bg =  cl.get_pos_neg_scores(indices_adapt_bg,
                                                                            pattern_times,
                                                                            scores_adapt_bg,
                                                                            classifier.window[1])
            positives_coarse, negatives_coarse =  cl.get_pos_neg_scores(indices_coarse,
                                                                        pattern_times,
                                                                        coarse_scores,
                                                                        classifier.window[1])
            all_positives.extend(positives)
            all_negatives.extend(negatives)
            all_positives_adapt_bg.extend(positives_adapt_bg)
            all_negatives_adapt_bg.extend(negatives_adapt_bg)
            all_positives_coarse.extend(positives_coarse)
            all_negatives_coarse.extend(negatives_coarse)
        else:
            break

aar_roc,aar_roc_vals = cl.get_roc(np.sort(all_positives)[::-1],
                                  np.sort(all_negatives)[::-1],num_frames)
aar_roc_adapt_bg,aar_roc_vals_adapt_bg = cl.get_roc(np.sort(all_positives_adapt_bg)[::-1],
                                                    np.sort(all_negatives_adapt_bg)[::-1],num_frames)
aar_roc_coarse,aar_roc_vals_coarse = cl.get_roc(np.sort(all_positives_coarse)[::-1],
                                                np.sort(all_negatives_coarse)[::-1],num_frames)







    liy_roc,liy_roc_vals = get_roc(np.sort(all_positives)[::-1],
                        np.sort(all_negatives)[::-1],
                        num_frames)

