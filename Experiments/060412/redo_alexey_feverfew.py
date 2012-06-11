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

texp = template_exp.\
    Experiment(patterns=[np.array(('aa','r')),np.array(('ah','r'))],
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=abst_threshold,
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7,
               offset=3
               )
train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)

# now we get the object spectrograms just as is done in Alexey with the 

train_data_iter.reset_exp()
train_specs = []
cur_iter = 0

while train_data_iter.next(wait_for_positive_example=True,
                           compute_S=True,compute_patterns_specs=True,compute_E=False):
    if cur_iter % 10 == 0:
        print cur_iter
    train_specs.extend(train_data_iter.patterns_specs)
    cur_iter += 1

ts = np.empty(len(train_specs),dtype=object)
for t in xrange(len(train_specs)): ts[t] = train_specs[t]

# save the training spectrograms
np.save('ts_aar060712',ts)


# going to compute the mean length
mean_length = int(round(np.mean([t.shape[1] for t in train_specs])))
# mean_length = 40

train_E = []
for t in xrange(len(train_specs)):
    E, edge_feature_row_breaks, edge_orientations = esp._edge_map_no_threshold(train_specs[t])  
    esp.threshold_edgemap(E,
                          .7,
                          edge_feature_row_breaks,
                          report_level=False,)
    esp.spread_edgemap(E,
                       edge_feature_row_breaks,
                       edge_orientations,
                       spread_length=2)
    train_E.append(E)

template_height = train_E[0].shape[0]

registered_examples = et._register_all_templates(len(train_E),
                            template_height,
                            mean_length,
                              train_E)

np.save('registered_aar060712',registered_examples)

aar_template = np.mean(registered_examples,axis=2)

np.save('aar_template060612',aar_template)

# BTT = .5
aar_salient = aar_template > .5
np.save('aar_salient060612',aar_salient)

def j0Detect(E,aar_salient):
    num_detections = E.shape[1] - aar_salient.shape[1]
    Responses = -np.inf * np.ones(E.shape[1])
    for t in xrange(num_detections):
        Responses[t] = np.sum(E[:,t:t+aar_salient.shape[1]][aar_salient])
    return Responses

def detect(responses,threshold):
    detections = np.empty(responses.shape[0],dtype=bool)
    detections[:] = False
    detections[1:-1] = np.logical_and(
        responses[1:-1] >= threshold,
        np.logical_or(
            np.logical_and(
                responses[1:-1] > responses[2:],
                responses[1:-1] >= responses[:-2]),
            np.logical_and(
                responses[1:-1] >= responses[2:],
                responses[1:-1] > responses[:-2])))
    return detections

tune_data_iter.reset_exp()    
c0 = 0
positive_J0_stats = []
cur_iter=0
while tune_data_iter.next(wait_for_positive_example=True,
                           compute_E=True,compute_pattern_times=False):
    if cur_iter % 10 == 0:
        print cur_iter
    tune_data_iter.pattern_times = esp.get_pattern_times(tune_data_iter.patterns,
                                              tune_data_iter.phns,
                                              tune_data_iter.feature_label_transitions)
    esp._edge_map_threshold_segments(tune_data_iter.E,
                                 aar_template.shape[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = tune_data_iter.edge_orientations,
                                 edge_feature_row_breaks = tune_data_iter.edge_feature_row_breaks)
    responses = j0Detect(tune_data_iter.E,aar_salient)
    detections = detect(responses,c0)
    for time_pair in tune_data_iter.pattern_times:
        positive_J0_stats.append(responses[time_pair[0]-mean_length/3:time_pair[1]][detections[time_pair[0] -mean_length/3:time_pair[1]]])
    cur_iter += 1

J0_vals = sorted(map(np.max,positive_J0_stats))


np.save('tune_E_060612',tune_data_iter.E)

texp_test = template_exp.\
    Experiment(patterns=[np.array(('aa','r')),np.array(('ah','r'))],
               data_paths_file=root_path+'Data/WavFilesTestPaths_feverfew',
               spread_length=1,
               abst_threshold=abst_threshold,
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7,
               offset=3
               )
test_data_iter, _ =\
    template_exp.get_exp_iterator(texp_test)


pos_detect = []
neg_detect = []
while tune_data_iter.next(wait_for_positive_example=True,
                           compute_E=True,compute_pattern_times=False):
    if cur_iter % 10 == 0:
        print cur_iter
    tune_data_iter.pattern_times = esp.get_pattern_times(tune_data_iter.patterns,
                                              tune_data_iter.phns,
                                              tune_data_iter.feature_label_transitions)
    esp._edge_map_threshold_segments(tune_data_iter.E,
                                 aar_template.shape[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = tune_data_iter.edge_orientations,
                                 edge_feature_row_breaks = tune_data_iter.edge_feature_row_breaks)
    responses = j0Detect(tune_data_iter.E,aar_salient)
    detections = detect(responses,c0)
    for time_pair in tune_data_iter.pattern_times:
        positive_J0_stats.append(responses[time_pair[0]-mean_length/3:time_pair[1]][detections[time_pair[0] -mean_length/3:time_pair[1]]])
    cur_iter += 1


train_data_iter.reset_exp()
avg_E = template_exp.AverageBackground()
while train_data_iter.next():
    avg_E.add_frames(train_data_iter.E,train_data_iter.edge_feature_row_breaks,
                     train_data_iter.edge_orientations,abst_threshold)
    if train_data_iter.cur_data_pointer % 10 == 0:
        print train_data_iter.cur_data_pointer


bg = avg_E.E.copy()

bg = np.minimum(np.maximum(bg,.1),.4)
reload(cl)

aar_template = np.minimum(np.maximum(aar_template,.01),.99)
aar_classifier = cl.Classifier(aar_template,coarse_factor=1,coarse_template_threshold = .5,bg=bg)

like_roc, count_roc = cl.get_roc_generous(test_data_iter, aar_classifier,coarse_thresh=-np.inf,
                   allowed_overlap = .1,
            edge_feature_row_breaks= tune_data_iter.edge_feature_row_breaks,
            edge_orientations=tune_data_iter.edge_orientations,
            abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02]),
            spread_radius=1)

coarse_thresh = 200
edge_feature_row_breaks= tune_data_iter.edge_feature_row_breaks
edge_orientations=tune_data_iter.edge_orientations
abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

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
         

#  we're pickling the test data so that way we can test on the same data
pkl_out = open(root_path+'Experiments/060412/redo_alexey_test_data_iter.pkl','wb')
cPickle.dump(test_data_iter,pkl_out)
pkl_out.close()
