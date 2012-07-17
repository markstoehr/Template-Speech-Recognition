"""
Copyright 2012
Author: Mark Stoehr

Here we perform a phone classification test of the phones
in timit, we use the Lee Hon list

"""

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
train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)


train_data_iter.reset_exp()
phn_set = set()
while train_data_iter.next():
    phn_set.update(train_data_iter.phns)

"""
List of all the phns
>>> sorted(list(phn_set))
['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
"""
    
phn_list = np.array(sorted(list(phn_set)))
np.save('phn_list070212',phn_list)

# timit mapping
leehon_mapping = { 'aa':'aa',
                   'ao':'aa',
                   'ax':'ah',
                   'ax-h':'ah',
                   'axr': 'er',
                   'hv':'hh',
                   'ix':'ih',
                   'el':'l',
                   'em':'m',
                   'en':'n',
                   'nx':'n',
                   'eng':'ng',
                   'zh':'sh',
                   'ux':'uw',
                   'plc':'sil',
                   'tcl':'sil',
                   'kcl':'sil',
                   'bcl':'sil',
                   'dcl':'sil',
                   'gcl':'sil',
                   'h#':'sil',
                   'pau':'sil',
                   'epi':'sil',
                   'sp':'sil'}

for phn in phn_list:
    if phn not in leehon_mapping.keys():
        leehon_mapping[phn] = phn


# q is discarded from the mapping
leehon_mapping['q'] = None

# we train a model for each phn
# in particular we are going to train a mixture
# model and do cross validation over the mixtures
# then do a final test
# we will use the same training and cross-validation division for each phone
for phn in phn_list:
    train_data_iter.reset_exp()
    datum_id = 0
    patterns = []
    lens = []
    offset = 3
    train_data_iter.patterns = [np.array((phn,))]
    while train_data_iter.next(wait_for_positive_example=True,
                               compute_pattern_times=True):
        if datum_id % 20 == 0:
            print datum_id
        datum_id += 1
        esp._edge_map_threshold_segments(train_data_iter.E,
                                 40,
                                 1, 
                                 threshold=.3,
                                 edge_orientations = train_data_iter.edge_orientations,
                                 edge_feature_row_breaks = train_data_iter.edge_feature_row_breaks)
        pattern_times = esp.get_pattern_times([np.array((phn,))],
                                              train_data_iter.phns,
                                              train_data_iter.feature_label_transitions)
        for p in pattern_times:
            patterns.append(train_data_iter.E[:,max(0,p[0]-offset):min(train_data_iter.E.shape[1],p[1]+offset)].copy())
            lens.append(p[1] - p[0] + 1)
    # get mean length
    mean_length = int(np.mean(np.array(lens)))
    template_height,template_length,registered_examples, template = et.simple_estimate_template(patterns,template_length=mean_length)
    np.save(phn+'_registered_examples070212',registered_examples)
    np.save(phn+'_template070212',template)


import template_speech_rec.bernoulli_em as bem
import template_speech_rec.classification as cl
   
num_mix_list = [2,3,4,6,9]    
for phn in phn_list:
    print phn
    registered_examples = np.load(phn+'_registered_examples070212.npy')
    for num_mix in num_mix_list:
        print num_mix
        bm = bem.Bernoulli_Mixture(num_mix,registered_examples)
        bm.run_EM(.00001)
        np.save(phn+str(num_mix)+'mix070412',bm.templates)


# get a universal stored background
# do mixture model for this later
stored_bg_avg_bg = template_exp.AverageBackground()
train_data_iter.reset_exp()
while train_data_iter.next():
    stored_bg_avg_bg.add_frames(train_data_iter.E,
                                train_data_iter.edge_feature_row_breaks,
                                train_data_iter.edge_orientations,
                                abst_threshold)


stored_bg = np.minimum(.4,np.maximum(stored_bg_avg_bg.E,.1))
    

for phn_id in xrange(len(phn_list)):
    tune_data_iter.reset_exp()
    datum_id = 0
    scores = [ [] for p in xrange(len(phn_list)) ]
    offset = 3
    phn = phn_list[phn_id]
    print "For classifier_list loading", phn+'_template070212.npy', ' '.join([ phn+str(num_mix)+'mix070412.npy' for num_mix in num_mix_list])
    classifier_list = [cl.Classifier(np.load(phn+'_template070212.npy'))]+ [ cl.Classifier(list(np.load(phn+str(num_mix)+'mix070412.npy'))) for num_mix in num_mix_list]
    while tune_data_iter.next():
        if datum_id % 20 == 0:
            print datum_id
            if datum_id % 40 == 0:
                print tune_data_iter.phns
        esp._edge_map_threshold_segments(tune_data_iter.E,
                                 40,
                                 1, 
                                 threshold=.3,
                                 edge_orientations = train_data_iter.edge_orientations,
                                 edge_feature_row_breaks = train_data_iter.edge_feature_row_breaks)
        for test_phn_id in xrange(tune_data_iter.phns.shape[0]):
            test_phn = tune_data_iter.phns[test_phn_id]
            # what's the id of the phone in the master list?
            test_phn_id_list = np.arange(phn_list.shape[0])[phn_list == test_phn][0]
            # if this is the beginning silence, use stored background estimate
            if test_phn_id == 0:
                bg = stored_bg
            else:
                bg = np.minimum(.4,np.maximum(np.mean(test_phn_E,axis=1),.1))
            if test_phn_id + 1 < tune_data_iter.feature_label_transitions.shape[0]:
                test_phn_E = tune_data_iter.E[:,max(0,
                                                    tune_data_iter.feature_label_transitions[test_phn_id]-offset):
                                                    min(tune_data_iter.E.shape[1],
                                                        tune_data_iter.feature_label_transitions[test_phn_id+1]+offset)]
            else:
                test_phn_E = tune_data_iter.E[:,max(0,
                                                    tune_data_iter.feature_label_transitions[test_phn_id]-offset):]
            scores[test_phn_id_list].append( map( lambda classifier_fun: classifier_fun.score_register(test_phn_E,bg),
                                             classifier_list))
        datum_id += 1
    out = open(phn_list[phn_id]+'_tune_scores070512.pkl','wb')
    cPickle.dump(scores,out)
    out.close()

#
#
#
#  Now run the same experiment where we are doing one-vs.all classification
#  this time we are going to a numpy array since we know how many scores there
#  are so we know the dimensionality of everything
#



#
#
#  code for analyzing the tuning experiment
# for each phone we want to compute the figure of merit score
#  the figure of merit score is the percent we classify correctly
#  its really just an area under the curve, we also want to know what errors are being made
#  with the other phones

# phn_confusion
# each row corresponds to a phone whose examples we look at in the timit database, the columns count the number of times
# a phone classifier gave the max score, the third dimension is the mixture component


# we want to estimate a bias term for each phone model since this will make classification more sane!
# in order to do this we need to perform the optimization over all the classification problems at once
# we are only going to have a row for each 


#
# tuning_examples_classifier_scores is where are the different classifier results are being stored
# and it is using this array that the bias estimation and the model selection will be performed jointly
# namely we need to figure out the correct bias term as well as the correct number of components to
# use for recognition
#

# variable keeps track of how many tuning examples
# there are for each phn
phn_num_tuning_examples = np.zeros(len(phn_list),dtype=int)
tuning_examples_classifier_scores = [ [] for i in xrange(len(phn_list))]
for phn_id in xrange(len(phn_list)):
    phn_num_tuning_examples[phn_id] = len(scores[phn_id])
    tuning_examples_classifier_scores[phn_id] = np.zeros((phn_num_tuning_examples[phn_id],len(phn_list),len(num_mix_list)+1))


for classifier_phn_id in xrange(len(phn_list)):
    print phn_list[classifier_phn_id]
    out = open(phn_list[classifier_phn_id]+'_tune_scores070512.pkl','rb')
    scores= cPickle.load(out)
    out.close()
    for classified_phn_id in xrange(len(phn_list)):
        for example_id,example_scores in enumerate(scores[classified_phn_id]):
            if example_id % 100 == 0:
                print classified_phn_id, example_id
            tuning_examples_classifier_scores[classified_phn_id][example_id,classifier_phn_id] = np.array(example_scores)


#
#
# tunin_examples_classifier_scores [ classified phone index] [ examples, classifier phone index, model selection index]
#

# maps a classifier phone index to which model selection to use, which is in range(len(num_mix_list)+1)
classifier_model_pick = np.zeros(len(phn_list),dtype=np.uint8)
# this is an estimate for the bias, we simply pick a number that gets good classification results
classifier_model_bias = np.zeros((len(phn_list),len(num_mix_list)+1))

#
# now we optimize the model choice and the bias
# bias is initialized to 0
# model choice is initialized to the single mixture case
#
# now we need to be able to compute what the actual scoring would be for a current
# model selection and bias estimation, or rather we should have a function evaluation
# to see what the current score is
#
#
# we can use a passive aggressive method for this optimization
#
#
# then we need to do a gradient for the bias score

def current_zero_one_loss(tuning_examples_classifier_scores,classifier_model_pick,classifier_model_bias,verbose=False):
    num_phns = len(scores)
    num_models = tuning_examples_classifier_scores[0].shape[-1]
    classifier_model_select_mask = np.zeros((num_phns,num_models),dtype=bool)
    classifier_model_select_mask[np.arange(num_phns),classifier_model_pick] = True
    cur_loss = 0
    num_examples = 0
    confusion_matrix = np.zeros((num_phns,num_phns))
    for classified_phn_id in xrange(num_phns):
        if verbose:
            print "Working on performance for classifying: %d" % classified_phn_id
        for example_id, example in enumerate(tuning_examples_classifier_scores[classified_phn_id]):
            best_classifier_phn_id = np.argmax(example[classifier_model_select_mask] + classifier_model_bias[classifier_model_select_mask])
            if best_classifier_phn_id != classified_phn_id:
                cur_loss += 1
            confusion_matrix[classified_phn_id,best_classifier_phn_id] += 1
            num_examples += 1
    return cur_loss,num_examples, confusion_matrix


cur_loss,num_examples, confusion_matrix =current_zero_one_loss(tuning_examples_classifier_scores,classifier_model_pick,classifier_model_bias,verbose=True)


def current_zero_one_loss_leehon(tuning_examples_classifier_scores,classifier_model_pick,classifier_model_bias,
                                 phn_list,leehon_mapping,
                                 verbose=False):
    num_phns = len(scores)
    num_models = tuning_examples_classifier_scores[0].shape[-1]
    classifier_model_select_mask = np.zeros((num_phns,num_models),dtype=bool)
    classifier_model_select_mask[np.arange(num_phns),classifier_model_pick] = True
    cur_loss = 0
    num_examples = 0
    confusion_matrix = np.zeros((num_phns,num_phns))
    for classified_phn_id in xrange(num_phns):
        if verbose:
            print "Working on performance for classifying: %d" % classified_phn_id
        for example_id, example in enumerate(tuning_examples_classifier_scores[classified_phn_id]):
            best_classifier_phn_id = np.argmax(example[classifier_model_select_mask] + classifier_model_bias[classifier_model_select_mask])
            if leehon_mapping[phn_list[best_classifier_phn_id]] != leehon_mapping[phn_list[classified_phn_id]]:
                cur_loss += 1
            confusion_matrix[classified_phn_id,best_classifier_phn_id] += 1
            num_examples += 1
    return cur_loss,num_examples, confusion_matrix

cur_loss_lh,num_examples_lh, confusion_matrix_lh =current_zero_one_loss_leehon(tuning_examples_classifier_scores,classifier_model_pick,classifier_model_bias,
                                                                               phn_list,leehon_mapping,
                                                                               verbose=True)


for phn_id in xrange(len(phn_list)):
    phn_num_tuning_examples[phn_id] = len(scores[phn_id])
    



#
#
#  Rerunning the experiment where I don't stretch anything and instead do padding
#
#
#


for phn_id in xrange(len(phn_list)):
    tune_data_iter.reset_exp()
    datum_id = 0
    scores = [ [] for p in xrange(len(phn_list)) ]
    offset = 3
    phn = phn_list[phn_id]
    print "For classifier_list loading", phn+'_template070212.npy', ' '.join([ phn+str(num_mix)+'mix070412.npy' for num_mix in num_mix_list])
    classifier_list = [cl.Classifier(np.load(phn+'_template070212.npy'))]+ [ cl.Classifier(list(np.load(phn+str(num_mix)+'mix070412.npy'))) for num_mix in num_mix_list]
    while tune_data_iter.next():
        if datum_id % 20 == 0:
            print datum_id
            if datum_id % 40 == 0:
                print tune_data_iter.phns
        esp._edge_map_threshold_segments(tune_data_iter.E,
                                 40,
                                 1, 
                                 threshold=.3,
                                 edge_orientations = train_data_iter.edge_orientations,
                                 edge_feature_row_breaks = train_data_iter.edge_feature_row_breaks)
        for test_phn_id in xrange(tune_data_iter.phns.shape[0]):
            test_phn = tune_data_iter.phns[test_phn_id]
            # what's the id of the phone in the master list?
            test_phn_id_list = np.arange(phn_list.shape[0])[phn_list == test_phn][0]
            # if this is the beginning silence, use stored background estimate
            if test_phn_id == 0:
                bg = stored_bg
            else:
                bg = np.minimum(.4,np.maximum(np.mean(test_phn_E,axis=1),.1))
            if test_phn_id + 1 < tune_data_iter.feature_label_transitions.shape[0]:
                test_phn_E = tune_data_iter.E[:,max(0,
                                                    tune_data_iter.feature_label_transitions[test_phn_id]-offset):
                                                    min(tune_data_iter.E.shape[1],
                                                        tune_data_iter.feature_label_transitions[test_phn_id+1]+offset)]
            else:
                test_phn_E = tune_data_iter.E[:,max(0,
                                                    tune_data_iter.feature_label_transitions[test_phn_id]-offset):]
            scores[test_phn_id_list].append( map( lambda classifier_fun: classifier_fun.score_register(test_phn_E,bg),
                                             classifier_list))
        datum_id += 1
    out = open(phn_list[phn_id]+'_tune_scores070512.pkl','wb')
    cPickle.dump(scores,out)
    out.close()

