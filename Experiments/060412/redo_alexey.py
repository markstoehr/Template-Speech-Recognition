root_path = '/home/mark/projects/Template-Speech-Recognition/'

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
               data_paths_file=root_path+'Data/WavFilesTrainPaths',
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
    train_specs.append(train_data_iter.patterns_specs)
    cur_iter += 1
    
