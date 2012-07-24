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
classifier_list = [np.array([np.load(save_dir + phn+'_template070212.npy')])]+ [ np.load(save_dir + phn+str(num_mix)+'mix070412.npy') for num_mix in num_mix_list]
Ts = classifier_list[2]
bgs = np.load(data_dir+ target_phn+'_bgs.npy')
E_windows = np.load(data_dir + target_phn+'_examples.npy')
E_num_cols_array = np.load(data_dir+target_phn +'_lengths.npy')

Cs,detect_sums = fast_like_noblas.te_fl_handmult(Ts,bgs,E_windows,
                                                 E_num_cols_array.astype(np.int32),
                                                 14,
                                                 0)


phn_target_results = [ [[] for phn_id2 in xrange(phn_list.shape[0])] for phn_id in xrange(phn_list.shape[0])]
for phn_id, phn in enumerate(phn_list):
    print "classifying phone is ", phn_id
    classifier_list = [np.array([np.load(save_dir + phn+'_template070212.npy')])]+ [ np.load(save_dir + phn+str(num_mix)+'mix070412.npy') for num_mix in num_mix_list]
    for target_phn_id, target_phn in enumerate(phn_list):
        print "target phn is %s" % target_phn
        bgs = np.load(data_dir+ target_phn+'_bgs.npy')
        E_windows = np.load(data_dir + target_phn+'_examples.npy')
        E_num_cols_array = np.load(data_dir+target_phn +'_lengths.npy')
        for Ts in classifier_list:
            Cs,detect_sums = fast_like_noblas.te_fl_handmult(Ts,bgs,E_windows,
                                                 E_num_cols_array.astype(np.int32),
                                                 14,
                                                 0)
            ds = detect_sums + np.tile(Cs.reshape(Cs.shape[0],
                                                  Cs.shape[1],
                                                  1),
                                       (1,1,detect_sums.shape[2]))
            phn_target_results[phn_id][target_phn_id].append((np.max(ds,axis=2),np.argmax(ds,axis=2)))

out = open("phn_target_results_init.pkl",'wb')
cPickle.dump(phn_target_results,out)
out.close()

plosive_test_case = ['p','t','k','b','d','g']


phn_target_results = [ [[] for phn_id2 in xrange(phn_list.shape[0])] for phn_id in xrange(phn_list.shape[0])]

for phn in plosive_test_case:
    phn_id = np.arange(61)[phn_list ==phn]
    print "classifying phone is ", phn_id, phn
    classifier_list = [np.array([np.load(save_dir + phn+'_template070212.npy')])]+ [ np.load(save_dir + phn+str(num_mix)+'mix070412.npy') for num_mix in num_mix_list]
    for target_phn in plosive_test_case:
        target_phn_id = np.arange(61)[phn_list ==target_phn]
        print "target phn is %s %d" % (target_phn,
                                       target_phn_id)
        bgs = np.load(data_dir+ target_phn+'_bgs.npy')
        E_windows = np.load(data_dir + target_phn+'_examples.npy')
        E_num_cols_array = np.load(data_dir+target_phn +'_lengths.npy')
        for Ts in classifier_list:
            Cs,detect_sums = fast_like_noblas.te_fl_handmult(Ts,bgs,E_windows,
                                                 E_num_cols_array.astype(np.int32),
                                                 14,
                                                 0)
            ds_idx = np.argmax(detect_sums,axis=2)
            ds =np.max(detect_sums,axis=2) + Cs
            ds_mix_idx = np.argmax(ds,axis=0)
            ds_mix = np.max(ds,axis=0)
            ds_time_idx = ds_idx[ds_mix_idx,np.arange(len(ds_mix))]
            phn_target_results[phn_id][target_phn_id].append(((ds_time_idx, ds_mix_idx, ds_mix )))



out = open("phn_target_results.pkl",'wb')
cPickle.dump(phn_target_results,out)
out.close()

out = open("plosive_idx.pkl",'wb')
cPickle.dump(plosive_idx, out)
out.close()

plosive_idx = [43,50,34,9,12,25]

use_model = [0,0,0,0,0,0]

def get_correct_incorrect(plosive_idx,phn_target_results,
                          cur_idx, use_model):
    score_pairs = []
    for target_phn_id, target_phn_big_id in enumerate(plosive_idx):
        other_scores = 
        phn_target_results[


phn_target_results[43][43]
pc = {}
for pl_id, pl1 in enumerate(plosive_idx[:-1]):
    for pl2 in plosive_idx[pl_id+1:]:
        print pl1, pl2
        pc[(pl1,pl2)] = (np.zeros((6,6)), np.zeros((6,6)))
        
