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



#
# Experiment for p,t,k classification, do vowels later
#
#
#

# make a training directory in the data folder

train_data_path = root_path + 'Data/Train/'

# load in the phone list to save everything to there

phn_list = np.load(root_path+'Experiments/070212/phn_list070212.npy')

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])



target_phn_list = ['p','t','k']
for phn in target_phn_list:
    phn_lengths = np.load(root_path+'Data/Train/'+phn+'_lengths.npy')
    phn_max_length = np.max(phn_lengths)
    example_id2utt_num = np.load(root_path+'Data/Train/'+phn+'_exampleid2utt_num.npy')
    E = np.load(root_path+'Data/Train/thresh_E_train0.npy')
    last_utt_num = -1
    phn_examples= np.empty((phn_lengths.shape[0],
                               E.shape[0],
                               phn_max_length),dtype = np.uint8)
    cur_utt_example = 0
    for example_id, use_utt_num in enumerate(example_id2utt_num):
        if use_utt_num > last_utt_num:
            cur_utt_example = 0
            E = np.load(root_path+'Data/Train/thresh_E_train' + str(use_utt_num)+'.npy')
            phns = np.load(root_path+'Data/Train/phns' + str(use_utt_num)+'.npy')
            feature_label_transitions = np.load(root_path+'Data/Train/feature_label_transitions' + str(use_utt_num)+'.npy')
        else:
            cur_utt_example += 1
        phn_idx = np.arange(phns.shape[0])[phns==phn][cur_utt_example]
        start_idx = feature_label_transitions[phn_idx]
        if start_idx + phn_max_length > E.shape[1]:
            phn_examples[example_id][:,:] = np.hstack((
                    E[:,start_idx: start_idx+phn_max_length],
                    np.tile(E[:,-1],(start_idx+phn_max_length - E.shape[1],1)).T))
        else:
            phn_examples[example_id][:,:] = E[:,start_idx: start_idx+phn_max_length]
        last_utt_num = use_utt_num
    np.save(root_path+'Data/Train/'+phn+'_examples.npy',phn_examples)

#
# look for pa, ta, ka
#
#
target_syll_list = (('p','aa'),('t','aa'),('k','aa'))
num_utts = 1293
for syll in target_syll_list:
    syll_lengths  = np.zeros(0,dtype=np.uint16)
    example_id2utt_num_pos = np.zeros((0,2),dtype=np.uint16)
    for utt_num in xrange(num_utts):
        phns = np.load(root_path+'Data/Train/phns' + str(utt_num)+'.npy')
        feature_label_transitions = np.load(root_path+'Data/Train/feature_label_transitions' + str(utt_num)+'.npy')
        syll_phn_length = len(syll)
        for phn in xrange(phns.shape[0]-syll_phn_length+1):
            if tuple(phns[phn:phn+syll_phn_length]) == syll:
                syll_lengths = np.append(
                    syll_lengths,
                    feature_label_transitions[phn+syll_phn_length]-
                    feature_label_transitions[phn])
                example_id2utt_num_pos = np.vstack(
                    (example_id2utt_num_pos,
                     (utt_num,feature_label_transitions[phn])))
        

        
    

# construct the basic experiment to see how mixtures, and, in particular
# mixtures of different lengths affect the classification rate

# load in the examples

phn = target_phn_list[0]

