#
#  Here we pick up on our other experiments and this time we apply
#  mixtures
#
#

# first we load in things
# to make this basic bit work
import numpy as np

root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle
sys.path.append(root_path)


# load in the parts that we use for coding

lower_cutoff=10
num_parts = 50
# retrieve the parts
parts = np.load(tmp_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))


# perform basic transformation so its easy to use
# convert to a smaller type for our cython functions
import template_speech_rec.get_train_data as gtrd


log_part_blocks, log_invpart_blocks = gtrd.reorg_parts_for_fast_filtering(parts)
log_part_blocks = log_part_blocks.astype(np.float32)
log_invpart_blocks = log_invpart_blocks.astype(np.float32)

# get the basic examples

aar_examples = np.load(tmp_data_path+'aar_examples.npy')
aar_lengths = np.load(tmp_data_path + 'aar_lengths.npy')
clipped_bgd = np.load(tmp_data_path+'clipped_train_bgd.npy')

import template_speech_rec.estimate_template as et

aar_template, aar_registered = et.register_templates_time_zero(aar_examples,aar_lengths,min_prob=.01)


#test_example_lengths = gtrd.get_detect_lengths(data_path+'Test/')


import template_speech_rec.bernoulli_mixture as bm


#
# we are now going to do the clustering procedure on these
#

aar_examples = np.load(tmp_data_path+'aar_examples.npy')
aar_lengths = np.load(tmp_data_path + 'aar_lengths.npy')
clipped_bgd = np.load(tmp_data_path+'clipped_train_bgd.npy')

aar_bgd_padded = et.pad_examples_bgd_samples(aar_examples,aar_lengths,clipped_bgd)


bem2 = bm.BernoulliMixture(2,aar_bgd_padded)
bem2.run_EM(.000001)

aar2 = et.recover_different_length_templates(bem2.affinities,aar_examples,aar_lengths)

for i in xrange(2):
    np.save(tmp_data_path + 'aar2_%d.npy' % i,aar2[i])

aar_mixture = tuple(
    np.load(tmp_data_path+'aar2_%d.npy' % i)
    for i in xrange(2))

# now we will get the detection array associated with these two

test_example_lengths = np.load(tmp_data_path+'test_example_lengths.npy')

detection_array = np.zeros((test_example_lengths.shape[0],
                            int(test_example_lengths.max()/float(log_part_blocks.shape[1]) + .5) + 2),dtype=np.float32)

linear_filters_cs = et.construct_linear_filters(aar_mixture,
                                             clipped_bgd)
# need to state the syllable we are working with
syllable = np.array(['aa','r'])


detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores_mixture(data_path+'Test/',                        
                                                                                         detection_array,
                                                                                         syllable,
                                                                                         linear_filters_cs,
                                                                                         log_part_blocks,
                                                                                         log_invpart_blocks,verbose=True)



np.save(tmp_data_path+'detection_array_aar_2.npy',detection_array)
out = open(tmp_data_path+'example_start_end_times_aar_2.pkl','wb')
cPickle.dump(example_start_end_times,out)
out.close()

out = open(tmp_data_path+'detection_lengths_aar_2.pkl','wb')
cPickle.dump(detection_lengths,out)
out.close()

import template_speech_rec.roc_functions as rf


window_start,window_end = rf.get_auto_syllable_window_mixture(aar_mixture)

max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        example_start_end_times,
                                                        detection_lengths,
                                                        window_start,
                                                        window_end)        


np.save(tmp_data_path+'aar_max_detect_vals_aar2_waliji.npy',max_detect_vals)

C0,C1 = rf.get_C0_C1_mixture(aar_mixture)

frame_rate = 1./5 * 1/.005

false_positive_rates, true_positive_rates = rf.get_roc_curve(max_detect_vals,
                        detection_array,
                        np.array(detection_lengths),
                        example_start_end_times,
                        C0,C1,frame_rate)

import matplotlib.pyplot as plt
plt.close()
x = true_positive_rates[true_positive_rates>=.75]
y = false_positive_rates[true_positive_rates>=.75]
plt.scatter(x,y )
plt.xlim((x.min()-.1,1.01))
plt.ylim((y.min()-.1,y.max()+.1))
plt.ylabel('Mistakes per second')
plt.xlabel('True Positive Rate')
plt.title('ROC curve for aar with 2 mixture components')
plt.savefig('../papers/AAR2_roc.png')


def compute_fom(true_positive_rates, false_positive_rates):
    main_detect_area = (false_positive_rates <= 10./360) * (false_positive_rates >= 1./360)
    return true_positive_rates[main_detect_area].sum()/main_detect_area.sum()

#In [160]: compute_fom(true_positive_rates, false_positive_rates)
#Out[160]: 0.45309568480300144


compute_fom(true_positive_rates, false_positive_rates)


plt.show()
