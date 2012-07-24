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

data_iter, _ =\
    template_exp.get_exp_iterator(texp,train_percent=1)



####

target_phn_list = ['p','t','k']
num_phns = {}
max_phn_length = {}
for phn in target_phn_list:
    num_phns[phn] = 0; max_phn_length[phn] = 0

esp._get_windows_signal(data_iter.s,data_iter.num_window_samples,data_iter.num_window_step_samples)

    print "Getting training examples for %d %s" % (phn_id, phn)
data_iter.reset_exp()
while data_iter.next():
    for phn_id, phn in enumerate(phn_list):
        if phn not in target_phn_list:
            continue
        # check that the utterance has the phn in question
        if phn not in data_iter.phns:
            continue
        phn_positions = np.arange(data_iter.phns.shape[0])[data_iter.phns == phn]
        num_phns[phn] += len(phn_positions)
        for phn_pos in phn_positions:
            max_phn_length[phn] = max(
                max_phn_length[phn],
                data_iter.feature_label_transitions[phn_pos+1] 
                - data_iter.feature_label_transitions[phn_pos])




            
for phn in target_phn_list:
    phn_examples = 2*np.ones((num_phns[phn],data_iter.E.shape[0],
                              max_phn_length[phn]),dtype=np.uint8)
    phn_bgs = np.empty((num_phns[phn], data_iter.E.shape[0]), dtype=np.float64)
    phn_lengths = np.empty(num_phns[phn],dtype=np.uint16)
    example_id2utt_num = np.empty(num_phns[phn],dtype=np.uint16)
    cur_example = 0
    if phn == 'p':
        data_iter.reset_exp()
        while data_iter.next():
            # threshold the edge map
            if data_iter.cur_data_pointer % 20 == 0: print data_iter.cur_data_pointer
            esp._edge_map_threshold_segments(data_iter.E,
                                         40,
                                         2, 
                                         threshold=.3,
                                         edge_orientations = data_iter.edge_orientations,
                                         edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            np.save(root_path+'Data/Train/thresh_E_train'
                    +str(data_iter.cur_data_pointer), data_iter.E)
            if data_iter.cur_data_pointer ==0 :
                np.save(root_path+'Data/Train/edge_feature_row_breaks', data_iter.edge_feature_row_breaks)
                np.save(root_path+'Data/Train/edge_orientations', data_iter.edge_orientations)
            np.save(root_path+'Data/Train/phns'
                    +str(data_iter.cur_data_pointer), data_iter.phns)
            np.save(root_path+'Data/Train/s'
                    +str(data_iter.cur_data_pointer), data_iter.s)
            np.save(root_path+'Data/Train/feature_label_transitions'
                   +str(data_iter.cur_data_pointer), data_iter.feature_label_transitions)
            # check that we are in an example that has the phone we care about
            if phn not in data_iter.phns: continue
            phn_positions = np.arange(data_iter.phns.shape[0])[data_iter.phns == phn]
            for phn_pos in phn_positions:
                end_idx_plus_one = data_iter.feature_label_transitions[phn_pos+1] 
                start_idx = data_iter.feature_label_transitions[phn_pos]
                phn_examples[cur_example][:,:end_idx_plus_one-start_idx] = data_iter.E[:,start_idx:end_idx_plus_one]
                example_id2utt_num[cur_example] = data_iter.cur_data_pointer
                phn_bgs[cur_example] = np.mean(data_iter.E[:,start_idx-20:start_idx])
                phn_lengths[cur_example] = end_idx_plus_one - start_idx
                cur_example += 1
                
    else:
        for utt_id in xrange(data_iter.num_data):
            phns = np.load(root_path+'Data/Train/phns'
                        +str(utt_id)+'.npy')
            if phn not in phns: continue
            E = np.load(root_path+'Data/Train/thresh_E_train'
                        +str(utt_id)+'.npy')
            feature_label_transitions = np.load(root_path+'Data/Train/feature_label_transitions'
                        +str(utt_id)+'.npy')
            phn_positions = np.arange(phns.shape[0])[phns == phn]
            for phn_pos in phn_positions:
                end_idx_plus_one = feature_label_transitions[phn_pos+1] 
                start_idx = feature_label_transitions[phn_pos]
                phn_examples[cur_example][:,:end_idx_plus_one-start_idx] = E[:,start_idx:end_idx_plus_one]
                example_id2utt_num[cur_example] = utt_id
                phn_bgs[cur_example] = np.mean(E[:,start_idx-20:start_idx])
                phn_lengths[cur_example] = end_idx_plus_one - start_idx
                cur_example += 1
    open(root_path+'Data/Train/num_data.txt','w').write(str(data_iter.num_data))
    np.save(root_path+'Data/Train/'+phn+'_examples.npy',phn_examples)
    np.save(root_path+'Data/Train/'+phn+'_bgs.npy',phn_bgs)
    np.save(root_path+'Data/Train/'+phn+'_lengths.npy',phn_lengths)
    np.save(root_path+'Data/Train/'+phn+'_exampleid2utt_num.npy',example_id2utt_num)

                

