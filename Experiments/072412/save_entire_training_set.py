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

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])



texp = template_exp.\
    Experiment(patterns=[np.array(('aa','r')),np.array(('ah','r'))],
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               #data_paths_file=root_path+'Data/WavFilesTrainPaths',
               spread_length=3,
               abst_threshold=abst_threshold,

               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7,
               offset=3
               )


# also going to save the phn list for later
# as well as trigrams and bigrams

from collections import defaultdict

unigrams = defaultdict()
bigrams = defaultdict()
trigrams = defaultdict()

data_iter, _ =\
    template_exp.get_exp_iterator(texp,train_percent=1.1)

while data_iter.next():
    np.save(train_data_path
            +str(data_iter.cur_data_pointer)+'phns.npy',
            data_iter.phns)
    E_spread = data_iter.E.copy()
    esp._edge_map_threshold_segments(E_spread,
                                         40,
                                         1, 
                                         threshold=.3,
                                         edge_orientations = data_iter.edge_orientations,
                                         edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
    np.save(train_data_path
            +str(data_iter.cur_data_pointer)+'E_spread3',
            E_spread.astype(np.uint8))
    del E_spread
    esp._edge_map_threshold_segments(data_iter.E,
                                         40,
                                         2, 
                                         threshold=.3,
                                         edge_orientations = data_iter.edge_orientations,
                                         edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
    np.save(train_data_path
            +str(data_iter.cur_data_pointer)+'E_spread5',
            data_iter.E.astype(np.uint8))
    np.save(train_data_path
            +str(data_iter.cur_data_pointer)
            +'feature_label_transitions',
            data_iter.feature_label_transitions)
    np.save(train_data_path
            +str(data_iter.cur_data_pointer)
            +'s.npy',data_iter.s)
    

np.save(train_data_path+'num_data',data_iter.num_data-1)


phn_list = set()

for data_id in xrange(data_iter.num_data):
    phn_list.update(np.load(train_data_path
                            +str(data_id+1)+'phns.npy'))


np.save(train_data_path + 'phn_list',np.array(phn_list))

from collections import defaultdict

phn_max_length = defaultdict(int)
phn_num_examples = defaultdict(int)


    
for data_id in xrange(1,data_iter.num_data-1):
    phns = np.load(train_data_path+str(data_id+1)
                   +'phns.npy')
    ftl = np.load(train_data_path+str(data_id+1)
                  +'feature_label_transitions.npy')
    for phn_id, phn in enumerate(phns):
        phn_num_examples[phn] += 1
        phn_max_length[phn] = max(phn_max_length[phn],
                                  ftl[phn_id+1]-ftl[phn_id])
        

out = open(train_data_path + 'phn_max_length.pkl','w')
cPickle.dump(phn_max_length,out)
out.close()
out = open(train_data_path + 'phn_num_examples.pkl','w')
cPickle.dump(phn_num_examples,out)
out.close()

out = open(train_data_path + 'phn_max_length.pkl','r')
phn_max_length = cPickle.dump(out)
out.close()
out = open(train_data_path + 'phn_num_examples.pkl','r')
cPickle.dump(phn_num_examples,out)
out.close()




target_phn_list = ['p','t','k']
for phn in phn_list:
    phn_examples_spread3 = np.empty((phn_num_examples[phn],
                                     data_iter.E.shape[0],
                                     phn_max_length[phn]),dtype=np.uint8)
   phn_examples_spread5 = np.empty((phn_num_examples[phn],
                                     data_iter.E.shape[0],
                                     phn_max_length[phn]),dtype=np.uint8) 
   phn_bgs = np.empty((phn_num_examples[phn],
                       data_iter.E.shape[0]))
   phn_lengths = np.empty(phn_num_examples[phn],np.uint8)
   phn_loc_stats = np.empty((phn_num_examples[phn],
                             4),dtype=np.intc)
   for data_id in xrange():
