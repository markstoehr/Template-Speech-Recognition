root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
#root_path = '/home/mark/Template-Speech-Recognition/'

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
phn_examples = np.load(root_path+'Data/'+phn+'_examples.npy')
phn_lengths = np.load(root_path+ 'Data/'+phn+'_lengths.npy')

unregistered_examples = [p[:,:l] for p,l in zip(phn_examples,phn_lengths)]

template_lengths= [6,7,8,9,10]
template_versions = []
for l in template_lengths:
    t_len, t_height, _ , template = et.simple_estimate_template(unregistered_examples,template_length=l)
    template_versions.append(template)



# classification output
# going to create a list of the experiments
# each thing is going to be error rates and need some explanation
# for the data provenance from the simulation
# 

#
#1 /p/ itself
# need to redo the feature extraction with the features being less spread
# it would make sense to do this with the whole training data, this time, simply let it run over night
# want to make sure that everything is in order


p_bgs = np.load(root_path+'Data/p_bgs.npy')

p_p_scores = np.zeros((phn_examples.shape[0],len(template_versions)))
for t_id, t in enumerate(template_versions):
    for p_id,p_ex in enumerate(phn_examples):
        p_p_scores[p_id,t_id] = sum(tt.score_template_background_section(t,p_bgs[p_id],p_ex[:,:t.shape[1]]))

# going to do a regression test based on length and look at
# the residuals from that
p_lengths_regress = np.vstack((np.ones(phn_lengths.shape[0]),
                               np.vstack((phn_lengths,
                                          phn_lengths**2)))).T

p_p_max_idx = np.argsort(p_p_scores, axis=1)

p_p_least_squares = np.linalg.solve(np.dot(p_lengths_regress.T ,
                                           p_lengths_regress),
                                    np.dot(p_lengths_regress.T,
                                           p_p_max_idx))



t_examples = np.load(root_path+'Data/t_examples.npy')
t_lengths = np.load(root_path+'Data/t_lengths.npy')
unregistered_examples = [p[:,:l] for p,l in zip(t_examples,t_lengths)]

template_versions_t = []
for l in template_lengths:
    t_len, t_height, _ , template = et.simple_estimate_template(unregistered_examples,template_length=l)
    template_versions_t.append(template)


p_examples = np.load(root_path+'Data/p_examples.npy')
t_p_scores = np.zeros((p_examples.shape[0],len(template_versions)))


p_examples = np.load(root_path+'Data/p_examples.npy')

p_bgs = np.load(root_path+'Data/p_bgs.npy')

p_p_scores = np.zeros((p_examples.shape[0],len(template_versions)))
for template_id, template in enumerate(template_versions_p):
    for p_id,p_ex in enumerate(p_examples):
        p_p_scores[p_id,template_id] = sum(tt.score_template_background_section(template,p_bgs[p_id],p_ex[:,:template.shape[1]]))

t_p_scores = np.zeros((p_bgs.shape[0],len(template_versions)))
for template_id, template in enumerate(template_versions_t):
    for p_id,p_ex in enumerate(p_examples):
        p_p_scores[p_id,template_id] = sum(tt.score_template_background_section(template,p_bgs[p_id],p_ex[:,:template.shape[1]]))

p_p_t_p_compare = np.argmax(np.vstack((np.max(p_p_scores,axis=1),
                             np.max(t_p_scores,axis=1))),axis=0)
 #23 mistakes - not bad
# want to see if there is a particular length distribution to the examples
#maybe something about the phones themselves
p_lengths = np.load(root_path+'Data/p_lengths.npy')
p_lengths[p_p_t_p_compare.astype(bool)]
#
# median is 6
# mean is 7 on those examples, so lengths aren't hurting us
# curious about the phonetic context
# example utt_num type question

# we'll now do the reverse experiment on t's
# this time we'll just take the mean lengths


