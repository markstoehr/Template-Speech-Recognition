#
# Experiment to see how different lengths affect the
# recognition ability, we also want to track where the errors are located
# inside the template, similarly, we want to understand where the improvement
# from the mixtures and how the mixtures can be used to help

root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)

#import template_speech_rec.data_loading as dl
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt


reload(template_exp)
texp = template_exp.\
    Experiment(pattern=np.array(('aa','r')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )




train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.7)

output = open('data_iter050712.pkl','wb')
cPickle.dump(train_data_iter,output)
cPickle.dump(tune_data_iter,output)
output.close()


all_patterns = []
E_avg = template_exp.AverageBackground()            
train_data_iter.spread_length = 5

for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                         get_patterns=True):
        all_patterns.extend(train_data_iter.patterns)
        E_avg.add_frames(train_data_iter.E,
                         train_data_iter.edge_feature_row_breaks,
                         train_data_iter.edge_orientations,
                         train_data_iter.abst_threshold)
    else:
        break


output = open('train_data_iter050712.pkl','wb')
cPickle.dump(train_data_iter,output)
output.close()

output = open('tune_data_iter050712.pkl','wb')
cPickle.dump(tune_data_iter,output)
output.close()

    

mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)



for pattern in xrange(len(all_patterns)):
    # do the thresholding
    esp.threshold_edgemap(all_patterns[pattern],.30,
                          train_data_iter.edge_feature_row_breaks,
                          report_level=False,
                          abst_threshold=train_data_iter.abst_threshold)
    esp.spread_edgemap(all_patterns[pattern],
                       train_data_iter.edge_feature_row_breaks,
                       train_data_iter.edge_orientations,spread_length=5)


template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns,
                                  template_length=33)

template_shape = np.array([template_height,template_length])
np.save('mean_template050612',mean_template)
np.save('template_shape050612',template_shape)


#########################################
#
#
# Testing the estimated templates
#
#

use_paths = tune_data_iter.paths[:]

reload(template_exp)
texp = template_exp.\
    Experiment(pattern=np.array(('aa','r')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )




train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.7)

train_data_iter2, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.7)


tune_data_iter.paths = use_paths[:]

#
# Run the detection experiment to get a sense of the effect that
# lengths have on the experimental results
# also use simpler template experiments code functions

all_patterns_context = []
tune_data_iter.spread_length = 3


#
# just testing out the code right now
#
#
datum_id = 0
has_pos = tune_data_iter.next(get_patterns_context=True)
sw = template_exp.SlidingWindowJ0(tune_data_iter.E,mean_template,
                                  quantile=.49)

#
# Original version of the code
#
#
#


tune_data_iter.reset_exp()

all_patterns_context = []
false_alarms = []
for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(compute_patterns_context=True,compute_pattern_times=True):
        all_patterns_context.extend(tune_data_iter.patterns_context)
        negative_examples = np.empty(tune_data_iter.E.shape[1],dtype=bool)
        negative_examples[:] = True
        for pattern_time in tune_data_iter.pattern_times:
            negative_examples[pattern_time[0]-template_length/3\
                                  :pattern_time[0]+template_length/3] = False
        neg_E = tune_data_iter.E[:,negative_examples]
        detections = get_detections(neg_E,mean_template,
                                    tune_data_iter.bg_len,
                                    mean_background,
                                    tune_data_iter.edge_feature_row_breaks,
                                    tune_data_iter.edge_orientations)
        false_alarms.extend(suppress_detections(detections,mean_template,
                                                tune_data_iter.phns,
                                                tune_data_iter.pattern,
                                                datum_id,
                                                tune_data_iter.feature_label_transitions,
                                                ))
    else:
        break


#
#
# New version of the code meant specifically to make this
# stuff easier
#
#
#


tune_data_iter.reset_exp()

all_patterns_context = []
false_alarms = []
for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(compute_patterns_context=True,compute_pattern_times=True):
        all_patterns_context.extend(tune_data_iter.patterns_context)
        false_alarms.extend(get_new_false_alarms(tune_data_iter.E,
                                                 tune_data_iter.phns,
                                                 tune_data_iter.pattern,
                                                 tune_data_iter.pattern_times,
                                                 mean_template,
                         tune_data_iter.bg_len,mean_background,
                         tune_data_iter.edge_feature_row_breaks,
                         tune_data_iter.edge_orientations,
                         tune_data_iter.feature_label_transitions,
                         datum_id,
                         ))
    else:
        break


#
#
#
# new version of the code that will enable
# me to set the j0 threshold so I don't have to muck around
# with all these detections
#
#
#

tune_data_iter.reset_exp()

all_patterns_context = []
false_alarms = []
for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                           compute_patterns_context=True,compute_pattern_times=True):
        all_patterns_context.extend(tune_data_iter.patterns_context)
        false_alarms.extend(get_new_false_alarms(tune_data_iter.E,
                                                 tune_data_iter.phns,
                                                 tune_data_iter.pattern,
                                                 tune_data_iter.pattern_times,
                                                 mean_template,
                         tune_data_iter.bg_len,mean_background,
                         tune_data_iter.edge_feature_row_breaks,
                         tune_data_iter.edge_orientations,
                         tune_data_iter.feature_label_transitions,
                         datum_id,
                         ))
    else:
        break




def get_new_false_alarms(E,phns,pattern,
                         pattern_times,template,
                         bg_len,bgd,
                         edge_feature_row_breaks,
                         edge_orientations,
                         label_transitions,
                         path_id,
                         ):
    template_length = template.shape[1]
    negative_examples = np.empty(E.shape[1],dtype=bool)
    negative_examples[:] = True
    for pattern_time in pattern_times:
        negative_examples[pattern_time[0]-template_length/3\
                              :pattern_time[0]+template_length/3] = False
    neg_E = E[:,negative_examples]
    detections = get_detections(neg_E,template,
                                bg_len,
                                bgd,
                                edge_feature_row_breaks,
                                edge_orientations)
    return suppress_detections(detections,
                               template,
                               phns,pattern,
                               label_transitions,path_id)


def get_detections(neg_E,mean_template,bg_len,mean_bgd,
                   edge_feature_row_breaks,edge_orientations,
                   spread_length=3,abst_threshold = .0001*np.ones(8)):
    num_detections = neg_E.shape[1] - mean_template.shape[1]
    detections = []
    for d in xrange(num_detections):
        E_segment = neg_E[:,d:d+mean_template.shape[1]].copy()
        esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)        
        if d > bg_len:
            bg = np.maximum(np.minimum(np.mean(neg_E[:,d-bg_len:d],
                                               axis=1),
                                       .4),
                            .01)
        else:
            bg = mean_bgd.copy()                            
        P,C =  tt.score_template_background_section(mean_template,
                                                    bg,E_segment)
        detections.append((P+C,d))
    return detections

def suppress_detections(detections,mean_template,phns,pattern,
                        label_transitions,path_id):
    # these are all the allowable detection points, they are removed as we go through the list of top points
    potential_detections = np.empty(len(detections),dtype=bool)
    potential_detections[:] = True
    # what is returned
    false_alarms = []
    detections.sort()
    for d in detections:
        if potential_detections[d[1]]:
            phn_id = np.sum(label_transitions<=d)
            false_alarms.append({"score": d[0],
                                 "frame": d[1],
                                 "path_id": path_id,
                                 "phn_context":phns[max(0,phn_id-1):
                                                        min(phns.shape[0],phn_id+pattern.shape[0]+1)],
                                 })
    return false_alarms
    
    false_alarms = []
    



#
#
#
#
#
#
#
#
#
