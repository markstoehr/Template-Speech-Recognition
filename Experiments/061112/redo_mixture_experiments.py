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


# get the mixture
import template_speech_rec.bernoulli_em as bem

registered_examples_aar = np.load(root_path + 'Experiments/053112/registered_aar060712.npy')

bm = bem.Bernoulli_Mixture(2,registered_examples_aar)

bm2 = bm
del bm
bm2.data_mat = root_path + 'Experiments/053112/registered_aar060712.npy'
pkl_out = open('bm_2_aar061112.pkl','wb')
cPickle.dump(bm2,pkl_out)
pkl_out.close()

#
# now we run the mixture experiment
#
#

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

bg = np.load(root_path + 'Experiments/053112/bg_060712.npy')

aar_classifier = cl.Classifier([bm2.templates[k] for k in xrange(bm2.templates.shape[0])],coarse_factor=1,coarse_template_threshold = .5,bg=bg)


data_iter = test_data_iter
if True:
    num_frames = 0
    all_positive_scores = []
    all_negative_scores = []
    all_positive_counts = []
    all_negative_counts = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1],
                          wait_for_positive_example=True):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            coarse_count_scores = -np.inf *np.ones(num_detections)
            coarse_scores = -np.inf * np.ones(num_detections)
            esp._edge_map_threshold_segments(data_iter.E,
                                 classifier.window[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = data_iter.edge_orientations,
                                 edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]]
                scores[d] = classifier.score_no_bg(E_segment)
                coarse_count_scores[d] = classifier.coarse_score_count(E_segment)
                if d>1 and d<num_detections-1:
                    if (coarse_count_scores[d-1] > coarse_thresh) and \
                            ((coarse_count_scores[d-1]>\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d-2]) or\
                                (coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>\
                                  coarse_count_scores[d-2]) ):
                        coarse_scores[d] = classifier.score_no_bg(E_segment)
            # get the positive and negative scores removed out
            pos_counts =[]
            pos_scores = []
            neg_indices = np.empty(scores.shape[0],dtype=bool)
            neg_indices[:]=True
            for pt in xrange(len(pattern_times)):
                pos_scores.append(np.max(scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                pos_counts.append(np.max(coarse_scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                neg_indices[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][1]] = False
            # get rid of overlapping instances
            neg_indices_counts_non_overlap = cl.remove_overlapping_examples(np.argsort(coarse_scores),
                                                        classifier.coarse_length,
                                                        int(allowed_overlap*classifier.coarse_length))
            neg_indices_non_overlap = cl.remove_overlapping_examples(np.argsort(scores),
                                                        classifier.window[1],
                                                        int(allowed_overlap*classifier.window[1]))
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_non_overlap] =True
            neg_indices_full = np.logical_and(neg_indices,neg_idx2)
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_counts_non_overlap] =True
            neg_indices_coarse = np.logical_and(neg_indices,neg_idx2)
            all_positive_scores.extend(pos_scores)
            all_positive_counts.extend(pos_counts)
            all_negative_scores.extend(scores[neg_indices_full])
            all_negative_counts.extend(coarse_scores[neg_indices_coarse])            
        else:
            break
    like_roc = cl.get_roc(np.sort(all_positive_scores)[::-1],
                        np.sort(all_negative_scores)[::-1],
                        num_frames)
    count_roc = cl.get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
         

np.save(root_path+'Experiments/061112/aar_mix2_like_roc061112',count_roc[1])
np.save(root_path+'Experiments/061112/aar_mix2_like_roc061112',like_roc[1])

del bm2

bm4 = bem.Bernoulli_Mixture(2,registered_examples_aar)


bm4.data_mat = root_path + 'Experiments/053112/registered_aar060712.npy'
pkl_out = open('bm_4_aar061112.pkl','wb')
cPickle.dump(bm4,pkl_out)
pkl_out.close()




aar_classifier = cl.Classifier([bm4.templates[k] for k in xrange(bm2.templates.shape[0])],coarse_factor=1,coarse_template_threshold = .5,bg=bg)


data_iter = test_data_iter
if True:
    num_frames = 0
    all_positive_scores = []
    all_negative_scores = []
    all_positive_counts = []
    all_negative_counts = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1],
                          wait_for_positive_example=True):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            coarse_count_scores = -np.inf *np.ones(num_detections)
            coarse_scores = -np.inf * np.ones(num_detections)
            esp._edge_map_threshold_segments(data_iter.E,
                                 classifier.window[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = data_iter.edge_orientations,
                                 edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]]
                scores[d] = classifier.score_no_bg(E_segment)
                coarse_count_scores[d] = classifier.coarse_score_count(E_segment)
                if d>1 and d<num_detections-1:
                    if (coarse_count_scores[d-1] > coarse_thresh) and \
                            ((coarse_count_scores[d-1]>\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d-2]) or\
                                (coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>\
                                  coarse_count_scores[d-2]) ):
                        coarse_scores[d] = classifier.score_no_bg(E_segment)
            # get the positive and negative scores removed out
            pos_counts =[]
            pos_scores = []
            neg_indices = np.empty(scores.shape[0],dtype=bool)
            neg_indices[:]=True
            for pt in xrange(len(pattern_times)):
                pos_scores.append(np.max(scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                pos_counts.append(np.max(coarse_scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                neg_indices[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][1]] = False
            # get rid of overlapping instances
            neg_indices_counts_non_overlap = cl.remove_overlapping_examples(np.argsort(coarse_scores),
                                                        classifier.coarse_length,
                                                        int(allowed_overlap*classifier.coarse_length))
            neg_indices_non_overlap = cl.remove_overlapping_examples(np.argsort(scores),
                                                        classifier.window[1],
                                                        int(allowed_overlap*classifier.window[1]))
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_non_overlap] =True
            neg_indices_full = np.logical_and(neg_indices,neg_idx2)
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_counts_non_overlap] =True
            neg_indices_coarse = np.logical_and(neg_indices,neg_idx2)
            all_positive_scores.extend(pos_scores)
            all_positive_counts.extend(pos_counts)
            all_negative_scores.extend(scores[neg_indices_full])
            all_negative_counts.extend(coarse_scores[neg_indices_coarse])            
        else:
            break
    like_roc = cl.get_roc(np.sort(all_positive_scores)[::-1],
                        np.sort(all_negative_scores)[::-1],
                        num_frames)
    count_roc = cl.get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
         
bm2 = bem.Bernoulli_Mixture(2,registered_examples_aar)


bm2.data_mat = root_path + 'Experiments/053112/registered_aar060712.npy'
pkl_out = open('bm_2_aar061212.pkl','wb')
cPickle.dump(bm2,pkl_out)
pkl_out.close()


aar_classifier = cl.Classifier([bm2.templates[k] for k in xrange(bm2.templates.shape[0])],coarse_factor=1,coarse_template_threshold = .5,bg=bg)



data_iter = test_data_iter
if True:
    num_frames = 0
    all_positive_scores = []
    all_negative_scores = []
    all_positive_counts = []
    all_negative_counts = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1],
                          wait_for_positive_example=True):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            coarse_count_scores = -np.inf *np.ones(num_detections)
            coarse_scores = -np.inf * np.ones(num_detections)
            esp._edge_map_threshold_segments(data_iter.E,
                                 classifier.window[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = data_iter.edge_orientations,
                                 edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]]
                scores[d] = classifier.score_no_bg(E_segment)
                coarse_count_scores[d] = classifier.coarse_score_count(E_segment)
                if d>1 and d<num_detections-1:
                    if (coarse_count_scores[d-1] > coarse_thresh) and \
                            ((coarse_count_scores[d-1]>\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d-2]) or\
                                (coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>\
                                  coarse_count_scores[d-2]) ):
                        coarse_scores[d] = classifier.score_no_bg(E_segment)
            # get the positive and negative scores removed out
            pos_counts =[]
            pos_scores = []
            neg_indices = np.empty(scores.shape[0],dtype=bool)
            neg_indices[:]=True
            for pt in xrange(len(pattern_times)):
                pos_scores.append(np.max(scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                pos_counts.append(np.max(coarse_scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                neg_indices[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][1]] = False
            # get rid of overlapping instances
            neg_indices_counts_non_overlap = cl.remove_overlapping_examples(np.argsort(coarse_scores),
                                                        classifier.coarse_length,
                                                        int(allowed_overlap*classifier.coarse_length))
            neg_indices_non_overlap = cl.remove_overlapping_examples(np.argsort(scores),
                                                        classifier.window[1],
                                                        int(allowed_overlap*classifier.window[1]))
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_non_overlap] =True
            neg_indices_full = np.logical_and(neg_indices,neg_idx2)
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_counts_non_overlap] =True
            neg_indices_coarse = np.logical_and(neg_indices,neg_idx2)
            all_positive_scores.extend(pos_scores)
            all_positive_counts.extend(pos_counts)
            all_negative_scores.extend(scores[neg_indices_full])
            all_negative_counts.extend(coarse_scores[neg_indices_coarse])            
        else:
            break
    like_roc = cl.get_roc(np.sort(all_positive_scores)[::-1],
                        np.sort(all_negative_scores)[::-1],
                        num_frames)
    count_roc = cl.get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
         

# testing whether I get the same results if I have two copies of aar

registered_examples_aar = np.load(root_path + 'Experiments/053112/registered_aar060712.npy')

T = np.minimum(np.maximum(np.mean(registered_examples_aar,axis=0),.05),.95)

pkl_in = open('../060412/redo_alexey_test_data_iter.pkl','rb')
test_data_iter = cPickle.load(pkl_in)
pkl_in.close()

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

bg = np.load(root_path + 'Experiments/053112/bg_060712.npy')

aar_classifier = cl.Classifier([T,T,T],coarse_factor=1,coarse_template_threshold = .5,bg=bg)

allowed_overlap = .1
coarse_thresh = -np.inf
data_iter = test_data_iter
if True:
    num_frames = 0
    all_positive_scores = []
    all_negative_scores = []
    all_positive_counts = []
    all_negative_counts = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=aar_classifier.window[1],
                          wait_for_positive_example=True):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - aar_classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            coarse_count_scores = -np.inf *np.ones(num_detections)
            coarse_scores = -np.inf * np.ones(num_detections)
            esp._edge_map_threshold_segments(data_iter.E,
                                 aar_classifier.window[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = data_iter.edge_orientations,
                                 edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+aar_classifier.window[1]]
                scores[d] = aar_classifier.score_no_bg(E_segment)
                coarse_count_scores[d] = aar_classifier.coarse_score_count(E_segment)
                if d>1 and d<num_detections-1:
                    if (coarse_count_scores[d-1] > coarse_thresh) and \
                            ((coarse_count_scores[d-1]>\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d-2]) or\
                                (coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>\
                                  coarse_count_scores[d-2]) ):
                        coarse_scores[d] = aar_classifier.score_no_bg(E_segment)
            # get the positive and negative scores removed out
            pos_counts =[]
            pos_scores = []
            neg_indices = np.empty(scores.shape[0],dtype=bool)
            neg_indices[:]=True
            for pt in xrange(len(pattern_times)):
                pos_scores.append(np.max(scores[pattern_times[pt][0]-int(np.ceil(aar_classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(aar_classifier.window[1]/3.))]))
                pos_counts.append(np.max(coarse_scores[pattern_times[pt][0]-int(np.ceil(aar_classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(aar_classifier.window[1]/3.))]))
                neg_indices[pattern_times[pt][0]-int(np.ceil(aar_classifier.window[1]/3.)):pattern_times[pt][1]] = False
            # get rid of overlapping instances
            neg_indices_counts_non_overlap = cl.remove_overlapping_examples(np.argsort(coarse_scores),
                                                        aar_classifier.coarse_length,
                                                        int(allowed_overlap*aar_classifier.coarse_length))
            neg_indices_non_overlap = cl.remove_overlapping_examples(np.argsort(scores),
                                                        aar_classifier.window[1],
                                                        int(allowed_overlap*aar_classifier.window[1]))
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_non_overlap] =True
            neg_indices_full = np.logical_and(neg_indices,neg_idx2)
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_counts_non_overlap] =True
            neg_indices_coarse = np.logical_and(neg_indices,neg_idx2)
            all_positive_scores.extend(pos_scores)
            all_positive_counts.extend(pos_counts)
            all_negative_scores.extend(scores[neg_indices_full])
            all_negative_counts.extend(coarse_scores[neg_indices_coarse])            
        else:
            break
    like_roc = cl.get_roc(np.sort(all_positive_scores)[::-1],
                        np.sort(all_negative_scores)[::-1],
                        num_frames)
    count_roc = cl.get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
         

##
#  The above tests whether the classification performance degrades if I have a list of the same classifier, it does not
#

np.save('one_mix_repeat_aar_like_roc061211',like_roc[1])

# Now we test whether deformation helps any by simply looking at a classifier with different lengths
#
#
#
reload(cl)

# construct the different length ones

T_front  = T[:,:T.shape[1]*.6666666666].copy()
T_back = T[:,T.shape[1]*.66666666:].copy()

T_short = np.zeros((T.shape[0],int(T.shape[1]*2./3)))
T_short_parts = np.vstack((
       (np.hstack((
                    T_front[:,:min(T_short.shape[1],T_front.shape[1])],
                    T_back[:,-max(T_short.shape[1]-T_front.shape[1],0):]))).reshape(1,T_short.shape[0],
                                                                               T_short.shape[1]),
       (np.hstack((
                    T_front[:,:max(T_short.shape[1]-T_back.shape[1],0)],
                    T_back[:,-min(T_short.shape[1],T_back.shape[1]):]))).reshape(1,T_short.shape[0],
                                                                               T_short.shape[1])))

T_short = np.mean(T_short_parts,axis=0)

T_long = np.hstack((T_front,T_back))

aar_classifier = cl.Classifier([T,T_short,T_long],coarse_factor=1,coarse_template_threshold = .5,bg=bg)
