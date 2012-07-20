root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
#root_path = '/home/mark/projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

# we are going to train a model fo reach phone as given below:
phns = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 'sp', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'sil']

phn_list = np.load(root_path+'Experiments/070212/phn_list070212.npy')


texp = template_exp.\
    Experiment(patterns=[np.array(('aa','r')),np.array(('ah','r'))],
               data_paths_file=root_path+'Data/WavFilesTestPaths_feverfew',
               #data_paths_file=root_path+'Data/WavFilesTrainPaths',
               spread_length=3,
               abst_threshold=abst_threshold,

               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7,
               offset=3
               )

test_data_iter, _ =\
    template_exp.get_exp_iterator(texp,train_percent=1)

data_dir = root_path + 'Data/Test/'
num_test_data =0
while test_data_iter.next():
    num_test_data+= 1
    np.save(data_dir+str(test_data_iter.cur_data_pointer)+'tune_E.npy',
            test_data_iter.E)
    np.save(data_dir+str(test_data_iter.cur_data_pointer)+'feature_label_transitions.npy',
            test_data_iter.feature_label_transitions)
    np.save(data_dir+str(test_data_iter.cur_data_pointer)+'phns.npy',
            test_data_iter.phns)
    np.save(data_dir+str(test_data_iter.cur_data_pointer)+
            's.npy', test_data_iter.s)


offset = 7

for phn_id in xrange(1):
    print phn_id, phn_list[phn_id]
    phn_tune_examples = []
    phn_tune_examples_bg = []
    for cur_data_pointer in xrange(num_tune_data):
        if cur_data_pointer % 20 == 0:
            print cur_data_pointer
        E = np.load(data_dir+str(cur_data_pointer)+'tune_E.npy')
        feature_label_transitions = np.load(data_dir+str(cur_data_pointer)+'feature_label_transitions.npy')
        seq_phns = np.load(data_dir+str(cur_data_pointer)+'phns.npy')
        for test_phn_id in xrange(seq_phns.shape[0]):
            test_phn = seq_phns[test_phn_id]
            # what's the id of the phone in the master list?
            test_phn_id_list = np.arange(phn_list.shape[0])[phn_list == test_phn][0]
            num_target_phns[test_phn_id_list] +=1
            if test_phn_id + 1 < feature_label_transitions.shape[0]:
                target_phns_max_length[test_phn_id_list] = max( target_phns_max_length[test_phn_id_list],
                                                            min(E.shape[1],feature_label_transitions[test_phn_id+1]+offset) - max(0,feature_label_transitions[test_phn_id]-offset))
            else:
                target_phns_max_length[test_phn_id_list] = max( target_phns_max_length[test_phn_id_list],
                                                            E.shape[1] - max(0,feature_label_transitions[test_phn_id]-offset))
    




