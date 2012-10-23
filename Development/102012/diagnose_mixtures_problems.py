# first we load in things
# to make this basic bit work
import numpy as np

root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
old_exp_path = root_path + 'Notebook/1/'
old_data_path = old_exp_path + 'data/'
exp_path = root_path + 'Notebook/2/'
tmp_data_path  = exp_path + 'data/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et


file_indices = np.load(tmp_data_path+'test_file_indices.npy')

num_mix = 3
i = 0

false_positive_rates =np.load(tmp_data_path+'fpr_aar_registered_new%d_%d.npy' % (num_mix,i))
true_positive_rates = np.load(tmp_data_path+'tpr_aar_registered_new%d_%d.npy' % (num_mix,i))

import matplotlib.pyplot as plt

colors = ['r','b','g','y','k','m','c']
markers = ['s','o','^','>','v','<','d','p']


plt.close()
plt.figure()

plt.xlim((.2,1.01))
plt.ylim((-.01,4.01))
for i in xrange(3):
    false_positive_rates =np.load(tmp_data_path+'fpr_aar_registered_new%d_%d.npy' % (num_mix,i))
    true_positive_rates = np.load(tmp_data_path+'tpr_aar_registered_new%d_%d.npy' % (num_mix,i))
    plt.scatter(true_positive_rates,false_positive_rates,c=colors[i],marker=markers[i])

plt.xlabel('True Positive Rate')
plt.ylabel('False Positives Per Second')
plt.show()

#
# we have confirmed that these are actually what we would hope
# to see. Going to see how the curves complement each other on an example

aar_templates = np.load(tmp_data_path+'aar_registered_templates_%d_%d.npy' % (num_mix,i))
out = open(tmp_data_path+'example_start_end_times_aar_%d_%d.npy' % (num_mix,i),'rb')
example_start_end_times = cPickle.load(out)
out.close()

detection_arrays = (,)
for i in xrange (num_mix):
    detection_arrays += (np.load(tmp_data_path+'detection_array_aar_new%d_%d.npy' % (num_mix,i)),)


clipped_bgd = np.load(tmp_data_path+'clipped_bgd_102012.npy')


aar_ts = ()
cs = ()
LFs = ()
for i in xrange(num_mix):
    aar_ts += (np.load(tmp_data_path+'aar_templates_%d_%d.npy' % (num_mix,i)),)
    LF,c = et.construct_linear_filter(aar_ts[i],
                                                        clipped_bgd)
    cs += (c,)
    LFs += (LF,)


# general plan
# example_start_end_times has a non-zero entry in the first utterance
# going to confirm that we have the right utterance

s = np.load(data_path + 'Test/'+file_indices[0]+'s.npy')
phns = np.load(data_path + 'Test/'+file_indices[0]+'phns.npy')
flts = np.load(data_path + 'Test/'+file_indices[0]+'feature_label_transitions.npy')

# don't seem to have the data on my local machine:
# somethign to check later
