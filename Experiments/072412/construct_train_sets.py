import numpy as np
from collections import defaultdict
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
exp_path = root_path+'Experiments/072412/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.estimate_template as et

out = open(data_path+'num_phn_examples.pkl','rb')
num_phn_examples = cPickle.load(out)
out.close()

class_array = np.load(data_path+'class_array.npy')

train_percent = .7
num_folds = 10
for phn_class in class_array:
    print phn_class
    num_examples = num_phn_examples[phn_class]
    train_masks = np.zeros((num_folds,
                            num_examples),
                           dtype=bool)
    for fold_id in xrange(num_folds):
        train_masks[fold_id][
            np.random.permutation(num_examples)[:int(.6*num_examples)]] = True
    np.save(exp_path+phn_class+'_train_masks',train_masks)
                            

def kmeans_linear(k,x):
    x.sort()
    numx = len(x)
    prev_assignment_idx = np.zeros(numx,dtype=np.uint8)
    assignment_idx = np.arange(numx) * k / numx
    assignments = np.zeros((k,numx),dtype=bool)
    assignments[assignment_idx,np.arange(numx)] = True
    k_vals = np.array([ np.mean(x[assignments[i]]) for i in xrange(k)])
    while np.all(prev_assignment_idx != assignment_idx):
        prev_assignment_idx = assignment_idx[:]
        assignment_idx = np.argmin((np.tile(x,(k,1)).T - np.tile(k_vals,(numx,1)))**2,axis=1)
        assignments[:] = False
        assignments[assignment_idx,np.arange(numx)] = True
        k_vals = np.array([ np.mean(x[assignments[i]]) for i in xrange(k)])
    return k_vals, assignment_idx
        
def kmedians_linear(k,x):
    x.sort()
    numx = len(x)
    prev_assignment_idx = np.zeros(numx,dtype=np.uint8)
    assignment_idx = np.arange(numx,dtype=np.uint8) * k / numx
    assignments = np.zeros((k,numx),dtype=bool)
    assignments[assignment_idx,np.arange(numx)] = True
    k_vals = np.array([ np.median(a[assignments[i]]) for i in xrange(k)])
    while np.all(prev_assignment_idx != assignment_idx):
        prev_assignment_idx = assignment_idx[:]
        assignment_idx = np.argmin(np.abs(np.tile(a,(k,1)).T - np.tile(k_vals,(numx,1))),axis=1)
        assignments[:] = False
        assignments[assignment_idx,np.arange(numx)] = True
        k_vals = np.array([ np.median(a[assignments[i]]) for i in xrange(k)])
    return k_vals, assignment_idx
        


    

for phn_class in class_array:
    print phn_class
    train_masks = np.load(exp_path+phn_class+'_train_masks.npy')
    for fold_id, train_mask in enumerate(train_masks):
        print fold_id
        train_phn_examples =  np.load(data_path+phn_class+"class_examples5.npy")
        train_phn_lengths = np.load(data_path+phn_class+"class_examples_lengths.npy")
        lengths = train_phn_lengths[train_mask].copy()
        template_length = int(np.mean(lengths)+.5)
        template_height,template_length,registered_templates, template  = et.simple_estimate_template(
            [t[:,:l] for t,l in zip(
                    train_phn_examples[train_mask],
                    lengths)],
            template_length=template_length)
        del registered_templates
        np.save(exp_path+phn_class+str(train_mask)+'template1_1_0',template)
        k_vals, assignment_idx = kmeans_linear(2,lengths)
        k_vals = list(frozenset((k_vals + .5).astype(np.uint8)))
        print k_vals
        for i in xrange(len(k_vals)):
            template_height,template_length,registered_templates, template  = et.simple_estimate_template([t[:,:l] for t,l,idx in zip(
                    train_phn_examples[train_mask],
                    lengths,
                    assignment_idx) if idx == i],template_length=k_vals[i])
            np.save(exp_path+phn_class+str(train_mask)+'template2_1_'+str(i),template)
        k_vals, assignment_idx = kmeans_linear(3,lengths)
        k_vals = list(frozenset((k_vals + .5).astype(np.uint8)))
        print k_vals
        for i in xrange(len(k_vals)):
            template_height,template_length,registered_templates, template  = et.simple_estimate_template([t[:,:l] for t,l,idx in zip(
                    train_phn_examples[train_mask],
                    lengths,
                    assignment_idx) if idx == i],template_length=k_vals[i])
            np.save(exp_path+phn_class+str(train_mask)+'template3_1_'+str(i),template)
        
