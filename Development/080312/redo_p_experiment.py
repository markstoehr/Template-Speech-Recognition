#
# Goal of this experiment is to find out why /p/ did so poorly compared to the other
# plosives
#
#

#
# couple of hypothese to test: should /pcl/ be included in the thing called '/p/'?
# want to see what context the errors might have been made in. Also want to test out
# Lian standardization method to see if I can't get thing working better

import numpy as np
from collections import defaultdict
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
exp_path = root_path+'Experiments/072412/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt

class_array = np.load(data_path+'class_array.npy')
plosives = ['p','t','k','b','d','g']
# 16 is as far as we have gotten in writing
# the classifier templates to memory
classifiers = list(class_array[1:16]) + plosives
# we have the three types of classifiers representing mixtures of 
# different lengths
num_classifier_models = 3

#construct a test run
plosive = plosives[0]
plosive_examples = np.load(data_path+plosive+'class_examples3.npy')
plosive_train_mask = np.load(exp_path+plosive+'_train_masks.npy')
plosive_lengths = np.load(data_path+plosive+'class_examples_lengths.npy')
plosive_bgs = np.load(data_path+plosive+'class_examples_bgs.npy')
num_validate = plosive_train_mask[0].shape[0]-np.sum(plosive_train_mask[0])
num_folds = plosive_train_mask.shape[0]
classifier_output = np.empty((num_folds,num_validate,
                              len(classifiers),num_classifier_models),
                             dtype=np.float32)
for classifier_id, classifier in enumerate(classifiers):
    print classifier
    # 1 template classification
    template0 = np.load(exp_path+classifier+'0template1_1_0.npy')
    t_len = template0.shape[1]
    valid_mask = np.logical_not(plosive_train_mask[0])
    classifier_output[0][:,classifier_id,0] = np.array([
        sum(tt.score_template_background_section(
            template0[:,:min(t_len,l)],
            bg,
            E[:,:min(t_len,l)]))/np.float32(min(t_len,l)) for E,l,bg in zip(plosive_examples[valid_mask],
                                                  plosive_lengths[valid_mask],
                                                  plosive_bgs[valid_mask])]).astype(np.float32)[:]
    # 2 template classification
    template0 = np.load(exp_path+classifier+'0template2_1_0.npy')
    template1 = np.load(exp_path+classifier+'0template2_1_1.npy')
    templates = (template0,template1)
    t_lens = (template0.shape[1], template1.shape[1])
    classifier_output[0][:,classifier_id,1] = np.array([
        max([sum(tt.score_template_background_section(
            template[:,:min(t_len,l)],
            bg,
            E[:,:min(t_len,l)]))/np.float32(min(t_len,l)) for template,t_len in zip(
                        templates,t_lens)]) 
            for E,l,bg in zip(plosive_examples[valid_mask],
                                                  plosive_lengths[valid_mask],
                                                  plosive_bgs[valid_mask])]).astype(np.float32)[:]
    template0 = np.load(exp_path+classifier+'0template3_1_0.npy')
    template1 = np.load(exp_path+classifier+'0template3_1_1.npy')
    template2 = np.load(exp_path+classifier+'0template3_1_2.npy')
    templates = (template0,template1,template2)
    t_lens = (template0.shape[1], template1.shape[1],template2.shape[1])
    classifier_output[0][:,classifier_id,2] = np.array([
        max([sum(tt.score_template_background_section(
            template[:,:min(t_len,l)],
            bg,
            E[:,:min(t_len,l)]))/np.float32(min(t_len,l)) for template,t_len in zip(
                        templates,t_lens)]) 
            for E,l,bg in zip(plosive_examples[valid_mask],
                                                  plosive_lengths[valid_mask],
                                                  plosive_bgs[valid_mask])]).astype(np.float32)[:]


classifier_output = np.empty((num_folds,num_validate,
                              len(classifiers),num_classifier_models),
                             dtype=np.float32)
for classifier_id, classifier in enumerate(classifiers):
    print classifier
    # 1 template classification
    template0 = np.load(exp_path+classifier+'0template1_1_0.npy')
    t_len = template0.shape[1]
    valid_mask = np.logical_not(plosive_train_mask[0])
    classifier_output[0][:,classifier_id,0] = np.array([
        sum(tt.score_template_background_section(
            template0[:,:min(t_len,l)],
            bg,
            E[:,:min(t_len,l)])) for E,l,bg in zip(plosive_examples[valid_mask],
                                                  plosive_lengths[valid_mask],
                                                  plosive_bgs[valid_mask])]).astype(np.float32)[:]
    # 2 template classification
    template0 = np.load(exp_path+classifier+'0template2_1_0.npy')
    template1 = np.load(exp_path+classifier+'0template2_1_1.npy')
    templates = (template0,template1)
    t_lens = (template0.shape[1], template1.shape[1])
    classifier_output[0][:,classifier_id,1] = np.array([
        max([sum(tt.score_template_background_section(
            template[:,:min(t_len,l)],
            bg,
            E[:,:min(t_len,l)])) for template,t_len in zip(
                        templates,t_lens)]) 
            for E,l,bg in zip(plosive_examples[valid_mask],
                                                  plosive_lengths[valid_mask],
                                                  plosive_bgs[valid_mask])]).astype(np.float32)[:]
    template0 = np.load(exp_path+classifier+'0template3_1_0.npy')
    template1 = np.load(exp_path+classifier+'0template3_1_1.npy')
    template2 = np.load(exp_path+classifier+'0template3_1_2.npy')
    templates = (template0,template1,template2)
    t_lens = (template0.shape[1], template1.shape[1],template2.shape[1])
    classifier_output[0][:,classifier_id,2] = np.array([
        max([sum(tt.score_template_background_section(
            template[:,:min(t_len,l)],
            bg,
            E[:,:min(t_len,l)])) for template,t_len in zip(
                        templates,t_lens)]) 
            for E,l,bg in zip(plosive_examples[valid_mask],
                                                  plosive_lengths[valid_mask],
                                                  plosive_bgs[valid_mask])]).astype(np.float32)[:]


