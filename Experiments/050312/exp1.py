#
# Experiment to see how different lengths affect the
# recognition ability, we also want to track where the errors are located
# inside the template, similarly, we want to understand where the improvement
# from the mixtures and how the mixtures can be used to help

root_path = '/home/mark/projects/Template-Speech-Recognition/'

import sys, os
sys.path.append(root_path)

#import template_speech_rec.data_loading as dl
import numpy as np
import template_speech_rec.template_experiments as template_exp

texp = template_exp.\
    Experiment(pattern=np.array(('aa','r')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )




train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.7)


all_patterns = []
E_avg = template_exp.AverageBackground()            
train_data_iter.spread_length = 5

for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    train_data_iter.next(wait_for_positive_example=True,
                         get_patterns=True)
    all_patterns.extend(train_data_iter.patterns)
    E_avg.add_frames(train_data_iter.E,
                     train_data_iter.edge_feature_row_breaks,
                     train_data_iter.edge_orientations,
                     train_data_iter.abst_threshold)
    


    dl.get_data_iterators(train=True,tune_percent=.3,
                          data_paths=root_path+'Data/TrainDataPaths.txt',
                          spread_length=3,abst_threshold=abst_threshold,
                          fft_length=512,num_window_step_samples=80,
                          freq_cutoff=3000,sample_rate=16000,
                          num_window_samples=320,kernel_length=7,
                          pattern=np.array(('aa','r'))

