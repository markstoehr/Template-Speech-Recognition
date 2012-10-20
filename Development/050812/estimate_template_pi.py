#root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
root_path = '/home/mark/projects/Template-Speech-Recognition/'

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
    Experiment(pattern=np.array(('p','iy')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )




train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.7)

output = open('data_iter_piy050912.pkl','wb')
cPickle.dump(train_data_iter,output)
cPickle.dump(tune_data_iter,output)
output.close()


all_patterns_context = []
all_patterns = []
E_avg = template_exp.AverageBackground()            
train_data_iter.spread_length = 5

for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_patterns_context=True,
                            compute_patterns=True,
                            max_template_length=50):
        # the context length is 11
        all_patterns_context.extend(train_data_iter.patterns_context)
        all_patterns.extend(train_data_iter.patterns)
        E_avg.add_frames(train_data_iter.E,
                         train_data_iter.edge_feature_row_breaks,
                         train_data_iter.edge_orientations,
                         train_data_iter.abst_threshold)
    else:
        break


output = open('all_patterns_piy050912.pkl','wb')
cPickle.dump(all_patterns,output)
output.close()

output = open('all_patterns_context_piy050912.pkl','wb')
cPickle.dump(all_patterns_context,output)
output.close()


mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)


template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns)

template_shape = np.array([template_height,template_length])
np.save('mean_template_piy050912',mean_template)
np.save('template_shape_piy050912',template_shape)

#
# Get the data for tuning the j0 threshold
#



#
# haven't run this stuff yet, but ran the stuff above
#

output = open('data_iter050812.pkl','rb')
train_data_iter=cPickle.load(output)
tune_data_iter = cPickle.load(output)
output.close()


    
tuning_patterns_context = []

for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                            compute_patterns_context=True,
                            max_template_length=40):
        # the context length is 11
        tuning_patterns_context.extend(tune_data_iter.patterns_context)
    else:
        break

output = open('tuning_patterns_context050912.pkl','wb')
cPickle.dump(tuning_patterns_context,output)
output.close()


#
# estimate the template using the
# patterns
#
output = open('all_patterns050912.pkl','rb')
all_patterns=cPickle.load(output)
output.close()

output = open('all_patterns_context050912.pkl','rb')
all_patterns_context = cPickle.load(output)
output.close()

#
#
# estimate the basic template, just going to use 2 parts
# for the time being start with the parts at just the half
# way point

