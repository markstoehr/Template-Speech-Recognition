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
    Experiment(patterns=[np.array(('aa','r'))],
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7,
               offset=3
               )
train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)

# now we get the object spectrograms just as is done in Alexey with the 

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

while tune_data_iter.next(wait_for_positive_example=True,
                           compute_E=True,compute_pattern_times=False,compute_E=False):
    if cur_iter % 10 == 0:
        print cur_iter
    pattern_times = esp.get_pattern_times(self.patterns,
                                              phns,
                                              feature_label_transitions))
    esp._edge_map_threshold_segments(tune_data_iter.E,
                                 aar_template.shape[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = tune_data_iter.edge_orientations,
                                 edge_feature_row_breaks = tune_data_iter.edge_feature_row_breaks)
    train_specs.extend(train_data_iter.patterns_specs)
    cur_iter += 1
