#
# Experiment to see how different lengths affect the
# recognition ability, we also want to track where the errors are located
# inside the template, similarly, we want to understand where the improvement
# from the mixtures and how the mixtures can be used to help

root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cpickle
sys.path.append(root_path)

#import template_speech_rec.data_loading as dl
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et


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
cpickle.dump(train_data_iter,output)
cpickle.dump(tune_data_iter,output)
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


#
# Run the detection experiment to get a sense of the effect that
# lengths have on the experimental results
# also use simpler template experiments code functions

all_patterns_context = []
tune_data_iter.spread_length = 3

for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                         get_patterns_context=True):
        all_patterns_context.extend(tune_data_iter.patterns_context)
        sw = SlidingWindow(tune_data_iter.E,mean_template,
                           )
        
        E_avg.add_frames(train_data_iter.E,
                         train_data_iter.edge_feature_row_breaks,
                         train_data_iter.edge_orientations,
                         train_data_iter.abst_threshold)
    else:
        break

    

