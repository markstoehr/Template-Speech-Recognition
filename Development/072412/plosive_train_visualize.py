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
import template_speech_rec.bernoulli_em as bem

data_dir = root_path + 'Data/'

p_examples = np.load(data_dir+'p_examples.npy')
p_lengths = np.load(data_dir+'p_lengths.npy')
import matplotlib.pyplot as plt

mean_length = np.mean(p_lengths)
pattern_examples = [p[:,:l] for p,l in zip(p_examples,p_lengths)]

plt.figure()
plt.show()
# look at the examples
for p in pattern_examples:
    plt.clf()
    raw_input('press Enter')
    plt.imshow(p,interpolation='nearest')

p_id=1
plt.figure()
for edge_id in xrange(8):
    plt.subplot(3,3,edge_id+1)
    plt.imshow(pattern_examples[p_id][edge_id*48:(edge_id+1)*48],
               interpolation='nearest')


def display_ex(pattern_examples,p_id):
    edge_stack = np.empty((48,0))
    for edge_id in xrange(8):
        edge_stack = np.hstack((edge_stack,np.hstack((2*np.ones((48,1)),
                                                  pattern_examples[p_id][
                        edge_id*48:
                            (edge_id+1)*48]))))
    plt.imshow(edge_stack,interpolation='nearest')
    plt.show()


# register the examples
template_height, template_length, registered_examples, template =  et.simple_estimate_template(pattern_examples,template_length=mean_length)

from pylab import imshow, plot, figure, show

def display_ex(example):
    edge_stack = np.empty((48,0))
    for edge_id in xrange(8):
        edge_stack = np.hstack((edge_stack,np.hstack((2*np.ones((48,1)),
                                                  example[
                        edge_id*48:
                            (edge_id+1)*48]))))
    plt.imshow(edge_stack,interpolation='nearest')
    plt.show()


t_examples = np.load(data_dir+'t_examples.npy')
t_lengths = np.load(data_dir+'t_lengths.npy')


mean_length = int(np.mean(t_lengths)+.25)
pattern_examples = [t[:,:l] for t,l in zip(t_examples,t_lengths)]
t_template_height, t_template_length, t_registered_examples, t_template =  et.simple_estimate_template(pattern_examples,template_length=mean_length)


k_examples = np.load(data_dir+'k_examples.npy')
k_lengths = np.load(data_dir+'k_lengths.npy')


mean_length = int(np.mean(k_lengths)+.25)
pattern_examples = [t[:,:l] for t,l in zip(k_examples,k_lengths)]
k_template_height, k_template_length, k_registered_examples, k_template =  et.simple_estimate_template(pattern_examples,template_length=mean_length)


# The templates don't look very different at all,
# I'm now going to try clustering them to see if I can make them look different
# on the basis of the clustering
# also I need to do the feature extraction that's the length of the longest
# example in the future, that part was messed up in my code yesterday

def k_median_clustering(k,vals):
    num_vals = len(vals)
    sorted_vals = sorted(vals)
    init = np.empty(k)
    for i in xrange(k):
        init[i] = sorted_vals[((i+1)*num_vals)/(k+1)]
    k_centers = init.copy()
    init = np.zeros(k)
    while np.all(init != k_centers):
        init[:] = k_centers[:]
        k_centers = k_median_iter(k,num_vals,k_centers,vals)
    return k_centers

def k_median_iter(k,num_vals,k_centers,vals):
    rep_vals = np.tile(vals,(k,1)).T
    assign_binary = np.zeros((num_vals,k),dtype=np.uint8)
    assign_binary[
        np.arange(num_vals),
        np.argmin(np.abs(np.tile(k_centers,(num_vals,1)) - 
                         rep_vals),axis=1)] = 1
    vals_rows = (assign_binary * rep_vals).T
    return np.array([np.median(vals_row[vals_row > 0],axis=0) for vals_row in vals_rows])
    
k_centers = k_median_iter(k,num_vals,k_centers,vals)
