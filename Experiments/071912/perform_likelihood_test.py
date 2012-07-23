root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
#root_path = '/home/mark/projects/Template-Speech-Recognition/'
fast_like_path = root_path + 'Experiments/071912/fast_like/'

import sys, os, cPickle
sys.path.append(root_path)
sys.path.append(fast_like_path)
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl
import fast_like.fast_like as fl


save_dir = root_path+'Experiments/070212/'

phn_list = np.load(save_dir + 'phn_list070212.npy')

num_mix_list = [2,3,4,6,9]    

data_dir = root_path + 'Data/Test/'

phn_id=0
phn = phn_list[phn_id]
target_phn_id = 0
target_phn = phn_list[target_phn_id]

Ts = classifier_list[2]
bgs = np.load(data_dir+ target_phn+'_bgs.npy')
E_windows = np.load(data_dir + target_phn+'_examples.npy')
E_num_cols_array = np.load(data_dir+target_phn +'_lengths.npy')
Cs,detect_sums = fl.templates_examples_fast_like(Ts,bgs,E_windows,
                                                 E_num_cols_array,
                                                 14,
                                                 0)



for phn_id, phn in enumerate(phn_list):
    classifier_list = [np.array([np.load(save_dir + phn+'_template070212.npy')])]+ [ np.load(save_dir + phn+str(num_mix)+'mix070412.npy') for num_mix in num_mix_list]
    for target_phn_id, target_phn in enumerate(phn_list):
        bgs = np.load(data_dir+ target_phn+'_bgs.npy')
        E_windows = np.load(data_dir + target_phn+'_examples.npy')
        E_num_cols_array = np.load(data_dir+target_phn +'_lengths.npy')
        for Ts in classifier_list:
            Cs,detect_sums = fl.templates_examples_fast_like(Ts,bgs,E_windows,
                                                             E_num_cols_array,
                                                             num_detections,
                                                             pad_front)


