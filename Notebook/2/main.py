# first we load in things
# to make this basic bit work
import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
old_exp_path = root_path + 'Notebook/1/'
old_data_path = old_exp_path + 'data/'
exp_path = root_path + 'Notebook/2/'
tmp_data_path  = exp_path + 'data/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
train_data_path = root_path+'Data/Train/'

file_indices = gtrd.get_data_files_indices(train_data_path)
syllable = np.array(['aa','r'])

clipped_bgd = np.load(old_data_path+'clipped_bgd_102012.npy')

padded_examples = np.load(old_data_path+'aar_padded_examples_bgd.npy')

lengths = np.load(old_data_path+'aar_lengths.npy')


test_example_lengths = np.load(old_data_path+'test_example_lengths_102012.npy')


detection_array = np.zeros((test_example_lengths.shape[0],
                            test_example_lengths.max() + 2),dtype=np.float32)

linear_filter,c = et.construct_linear_filter(aar_template,
                                             clipped_bgd)
# need to state the syllable we are working with
syllable = np.array(['aa','r'])


detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores(data_path+'Test/',
                                                                                         detection_array,
                                                                                         syllable,
                                                                                         linear_filter,c,verbose=True)

np.save(tmp_data_path+'detection_array_aar_1.npy',detection_array)
out = open(tmp_data_path+'example_start_end_times_aar_1.pkl','wb')
cPickle.dump(example_start_end_times,out)
out.close()

out = open(tmp_data_path+'detection_lengths_aar_1.pkl','wb')
cPickle.dump(detection_lengths,out)
out.close()

import template_speech_rec.roc_functions as rf


window_start,window_end = rf.get_auto_syllable_window(aar_template)

max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        example_start_end_times,
                                                        detection_lengths,
                                                        window_start,
                                                        window_end)
np.save(tmp_data_path+'max_detect_vals_aar2_waliji.npy',max_detect_vals)
C0,C1 = rf.get_C0_C1(aar_template)

frame_rate = 1/.005

false_positive_rates, true_positive_rates = rf.get_roc_curve(max_detect_vals,
                        detection_array,
                        np.array(detection_lengths),
                        example_start_end_times,
                        C0,C1,frame_rate)

np.save(tmp_data_path+'fpr_aar_1.npy',false_positive_rates)
np.save(tmp_data_path+'tpr_aar_1.npy',true_positive_rates)

# Now we are going to test the mixtures
# First thing to do is to test the registered mixtures
# then we are going to try the padded mixtures

import template_speech_rec.bernoulli_mixture as bm

bem = bm.BernoulliMixture(2,aar_registered)
bem.run_EM(.000001)



detection_array = np.zeros((test_example_lengths.shape[0],
                            test_example_lengths.max() + 2),dtype=np.float32)

linear_filters_cs = et.construct_linear_filters(bem.templates,
                                             clipped_bgd)


detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores_mixture(data_path+'Test/',
                                                                                         detection_array,
                                                                                         syllable,
                                                                                         linear_filters_cs,verbose=True)


np.save(tmp_data_path+'detection_array_aar_2.npy',detection_array)
out = open(tmp_data_path+'example_start_end_times_aar_2.pkl','wb')
cPickle.dump(example_start_end_times,out)
out.close()

out = open(tmp_data_path+'detection_lengths_aar_2.pkl','wb')
cPickle.dump(detection_lengths,out)
out.close()

import template_speech_rec.roc_functions as rf


window_start,window_end = rf.get_auto_syllable_window(bem.templates)

max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        example_start_end_times,
                                                        detection_lengths,
                                                        window_start,
                                                        window_end)
np.save(tmp_data_path+'max_detect_vals_aar2_koloy.npy',max_detect_vals)
C0,C1 = rf.get_C0_C1(aar_template)

frame_rate = 1/.005

false_positive_rates, true_positive_rates = rf.get_roc_curve(max_detect_vals,
                        detection_array,
                        np.array(detection_lengths),
                        example_start_end_times,
                        C0,C1,frame_rate)

np.save(tmp_data_path+'fpr_aar_2.npy',false_positive_rates)
np.save(tmp_data_path+'tpr_aar_2.npy',true_positive_rates)

for num_mix in [3,4,5,6]:
    print num_mix
    bem = bm.BernoulliMixture(num_mix,aar_registered)
    bem.run_EM(.000001)
    for i in xrange(num_mix):
        print i
        aar_template = bem.templates[i]
        np.save(tmp_data_path+'aar_templates_%d_%d.npy' % (num_mix,i),aar_template)
        detection_array = np.zeros((test_example_lengths.shape[0],
                                   test_example_lengths.max() + 2),dtype=np.float32)
        linear_filter,c = et.construct_linear_filter(aar_template,
                                                        clipped_bgd)
        detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores(data_path+'Test/',
                                                                                                       detection_array,
                                                                                                       syllable,
                                                                                                       linear_filter,c,verbose=True)
        np.save(tmp_data_path+'detection_array_aar_new%d_%d.npy' % (num_mix,i),detection_array)
        np.save(tmp_data_path+'aar_registered_templates_%d_%d.npy' % (num_mix,i),bem.templates)
        out = open(tmp_data_path+'example_start_end_times_aar_%d_%d.npy' % (num_mix,i),'wb')
        cPickle.dump(example_start_end_times,out)
        out.close()
        out = open(tmp_data_path+'detection_lengths_aar_%d_%d.npy' % (num_mix,i),'wb')
        cPickle.dump(detection_lengths,out)
        out.close()
        window_start,window_end = rf.get_auto_syllable_window(aar_template)
        max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        example_start_end_times,
                                                        detection_lengths,
                                                        window_start,
                                                        window_end)
        np.save(tmp_data_path+'max_detect_vals_aar%d_%d_koloy_new.npy' % (num_mix,i),max_detect_vals)
        C0,C1 = rf.get_C0_C1(aar_template)
        frame_rate = 1/.005
        false_positive_rates, true_positive_rates,detected_examples = rf.get_roc_curve(max_detect_vals,
                                                                     detection_array,
                                                                     np.array(detection_lengths),
                                                                     example_start_end_times,
                                                                     C0,C1,frame_rate,
                                                                     return_detected_examples=True)
        np.save(tmp_data_path+'fpr_aar_registered_new%d_%d.npy' % (num_mix,i),false_positive_rates)
        np.save(tmp_data_path+'tpr_aar_registered_new%d_%d.npy' % (num_mix,i),true_positive_rates)
        np.save(tmp_data_path+'detected_examples_aar_registered_new%d_%d.npy' % (num_mix,i),
                np.hstack(detected_examples))



import re,os

false_positive_rate_re = re.compile('fpr\_*')
true_positive_rate_re= re.compile('tpr\_*')
fpr_files = [f for f in os.listdir(tmp_data_path) if false_positive_rate_re.search(f)]
tpr_files = [f for f in os.listdir(tmp_data_path) if true_positive_rate_re.search(f)]

import matplotlib.pyplot as plt

colors = ['r','b','g','y','k','m','c']
markers = ['s','o','^','>','v','<','d','p']


plt.close()
plt.figure()

plt.xlim((.3,1.01))
plt.ylim((-.01,4.01))
scatter_plots = []
scatter_plot_names = []
for i in xrange(len(tpr)):
    tpr = np.load(tmp_data_path+tpr_files[i])
    fpr = np.load(tmp_data_path+fpr_files[i])
    p = plt.scatter(tpr,fpr,c=colors[i],marker=markers[i])
    scatter_plots.append(p)
    scatter_plot_names.append(tpr_files[i][len('tpr_'):-len('.npy')])

plt.xlabel('True Positive Rate')
plt.ylabel('False Positives Per Second')
plt.legend(scatter_plots,scatter_plot_names)
plt.show()





if __name__ == "__main__":
    main()
