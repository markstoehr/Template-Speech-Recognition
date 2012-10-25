import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092412/'
tmp_data_path = exp_path + 'data/'
scripts_path = exp_path + 'scripts/'
import sys, os, cPickle
sys.path.append(root_path)


##
# We assume that parts have already been computed
# we are not interested in comparing different part architectures at the moment
#

# load in the parts that we use for coding

lower_cutoff=10
num_parts = 50
# retrieve the parts
old_devel_data_path = root_path + 'Development/092412/data/'
parts = np.load(old_devel_data_path+'bm_templates%d_%d.npy' % (lower_cutoff,num_parts))

spec_parts = np.load(old_devel_data_path+'spec_avgs10k_%d_%d.npy' % (lower_cutoff,
                                                                      num_parts))

plt.close()
for i in range(10):
    for j in range(5):
        plt.subplot(10,5,i*5 + j)
        plt.imshow(spec_parts[i*5+j],cmap=cm.bone)
        plt.axis('off')
plt.show()

# perform basic transformation so its easy to use
# convert to a smaller type for our cython functions
import template_speech_rec.get_train_data as gtrd

log_part_blocks, log_invpart_blocks = gtrd.reorg_parts_for_fast_filtering(parts)
log_part_blocks = log_part_blocks.astype(np.float32)
log_invpart_blocks = log_invpart_blocks.astype(np.float32)

syllable = ('aa','r')

train_data_path = root_path+'Data/Train/'

file_indices = gtrd.get_data_files_indices(train_data_path)

avg_bgd, syllable_examples, backgrounds = gtrd.get_syllable_examples_backgrounds_files(train_data_path,
                                            file_indices,
                                            syllable,
                                            log_part_blocks,
                                            log_invpart_blocks,
                                            num_examples=-1,
                                            verbose=True)

np.save(tmp_data_path+'clipped_bgd_waliji.npy',np.clip(avg_bgd.E,
                                                       .1,
                                                       .4))

cur_tmp_data_path = root_path+'Notebook/5/data/'
waliji_bgd = np.load(cur_tmp_data_path+'clipped_bgd_waliji.npy')

out=open(tmp_data_path+'aar_waliji_syllable_examples.pkl','wb')
cPickle.dump(syllable_examples,out)
out.close()

out=open(cur_tmp_data_path+'aar_waliji_syllable_examples.pkl','rb')
syllable_examples = cPickle.load(out)
out.close()

parts_padded_examples, parts_lengths = et.extend_examples_to_max(waliji_bgd,syllable_examples,return_lengths=True)

np.save(cur_tmp_data_path+'parts_padded_examples.npy',parts_padded_examples)
np.save(cur_tmp_data_path+'parts_lengths.npy',parts_lengths)

test_example_lengths = gtrd.get_detect_lengths(test_file_indices,test_data_path)

parts_test_example_lengths = np.array(test_example_lengths)/5 + 1

import template_speech_rec.roc_functions as rf

mixture_vals = [2,3,4,5,6]
for num_mix in mixture_vals:
    print num_mix
    bem = bm.BernoulliMixture(num_mix,parts_padded_examples)
    bem.run_EM(.000001)
    parts_templates = et.recover_different_length_templates(bem.affinities,
                                                          parts_padded_examples,
                                                          parts_lengths)
    for i in xrange(num_mix):
        np.save(tmp_data_path+'parts_aar_template_%d_%d.npy' %(num_mix,i),parts_templates[i])
    detection_array = np.zeros((parts_test_example_lengths.shape[0],
                            parts_test_example_lengths.max() + 2),dtype=np.float32)
    linear_filters_cs = et.construct_linear_filters(parts_templates,
                                             waliji_bgd)
    for i in xrange(num_mix):
        np.save(cur_tmp_data_path+'linear_filter_aar_%d_%d.npy'%(num_mix,i),linear_filters_cs[i][0])
        np.save(cur_tmp_data_path+'c_aar_%d_%d.npy'%(num_mix,i),np.array(linear_filters_cs[i][0]))
    syllable = np.array(['aa','r'])
    detection_array,parts_example_start_end_times, parts_detection_lengths = gtrd.get_detection_scores_mixture(test_data_path,
                                                                                         detection_array,
                                                                                         syllable,
                                                                                         linear_filters_cs,
                                                                                                   log_part_blocks=log_part_blocks,
                                                                                                   log_invpart_blocks=log_invpart_blocks,
                                                                                                   verbose=True)
    np.save(cur_tmp_data_path+'parts_detection_array_aar_%d.npy' % num_mix,detection_array)
    if num_mix == 2:
        out = open(cur_tmp_data_path+'parts_example_start_end_times_aar.pkl','wb')
        cPickle.dump(parts_example_start_end_times,out)
        out.close()
        out = open(cur_tmp_data_path+'parts_detection_lengths_aar.pkl','wb')
        cPickle.dump(parts_detection_lengths,out)
        out.close()
    window_start = -2
    window_end = 2
    max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        parts_example_start_end_times,
                                                        parts_detection_lengths,
                                                        window_start,
                                                        window_end)
    np.save(cur_tmp_data_path+'parts_max_detect_vals_aar_%d.npy' % num_mix,max_detect_vals)
    C0 = 7
    C1 = int( 7 * 1.5 + .5)
    frame_rate = 1/.005 * 5
    fpr, tpr = rf.get_roc_curve(max_detect_vals,
                        detection_array,
                        np.array(parts_detection_lengths),
                        parts_example_start_end_times,
                        C0,C1,frame_rate)
    np.save(cur_tmp_data_path+'parts_fpr_aar_%d.npy' % num_mix,
            fpr)
    np.save(cur_tmp_data_path+'parts_tpr_aar_%d.npy' % num_mix,
            tpr)
    detection_clusters = rf.get_detect_clusters_threshold_array(max_detect_vals,
                                                                detection_array,
                                                                np.array(parts_detection_lengths),
                                                                C0,C1)
    out = open(cur_tmp_data_path+'parts_detection_clusters_aar_%d.npy' % num_mix,
               'wb')
    cPickle.dump(detection_clusters,out)
    out.close()

    


num_mix = 1
aar_parts_template, aar_registered = et.register_templates_time_zero(syllable_examples,min_prob=.01)
np.save(cur_tmp_data_path+'parts_aar_template_%d.npy' % num_mix,aar_parts_template)
detection_array = np.zeros((parts_test_example_lengths.shape[0],
                            parts_test_example_lengths.max() + 2),dtype=np.float32)
linear_filter,c = et.construct_linear_filter(aar_parts_template,
                                             waliji_bgd)
for i in xrange(num_mix):
    np.save(cur_tmp_data_path+'linear_filter_aar_%d_%d.npy'%(num_mix,i),linear_filter)
    np.save(cur_tmp_data_path+'c_aar_%d_%d.npy'%(num_mix,i),np.array(c))
syllable = np.array(['aa','r'])
detection_array,parts_example_start_end_times, parts_detection_lengths = gtrd.get_detection_scores(test_data_path,
                                                                                         detection_array,
                                                                                         syllable,
                                                                                         linear_filter,c,
                                                                                                   log_part_blocks=log_part_blocks,
                                                                                                   log_invpart_blocks=log_invpart_blocks,
                                                                                                   verbose=True)
np.save(cur_tmp_data_path+'parts_detection_array_aar_%d.npy' % num_mix,detection_array)
window_start = -2
window_end = 2
max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        parts_example_start_end_times,
                                                        parts_detection_lengths,
                                                        window_start,
                                                        window_end)
np.save(cur_tmp_data_path+'parts_max_detect_vals_aar_%d.npy' % num_mix,max_detect_vals)
num_mix = 1
max_detect_vals = np.load(cur_tmp_data_path+'parts_max_detect_vals_aar_%d.npy' % num_mix)
detection_array = np.load(cur_tmp_data_path+'parts_detection_array_aar_%d.npy' % num_mix)
C0 = 7
C1 = int( 7 * 1.5 + .5)
frame_rate = 1/.005 * 5
fpr, tpr = rf.get_roc_curve(max_detect_vals,
                        detection_array,
                        np.array(parts_detection_lengths),
                        parts_example_start_end_times,
                        C0,C1,frame_rate)
np.save(cur_tmp_data_path+'parts_fpr_aar_%d.npy' % num_mix,
            fpr)
np.save(cur_tmp_data_path+'parts_tpr_aar_%d.npy' % num_mix,
        tpr)
detection_clusters = rf.get_detect_clusters_threshold_array(max_detect_vals,
                                                                detection_array,
                                                                np.array(parts_detection_lengths),
                                                                C0,C1)
out = open(cur_tmp_data_path+'parts_detection_clusters_aar_%d.npy' % num_mix,
               'wb')
cPickle.dump(detection_clusters,out)
out.close()

import matplotlib.pyplot as plt
colors = ['r','b','g','y','k','m','c']
xomarkers = ['s','o','^','>','v','<','d','p']

mixture_vals = [1,2,4]
plt.close()
plt.figure()
plt.xlim((.2,1.1))
plt.ylim((-.1,3))
plt.xlabel('True Positive Rate')
plt.ylabel('False Positives Per Second')
for j,num_mix in enumerate(mixture_vals):
    fpr =np.load(cur_tmp_data_path+'parts_fpr_aar_%d.npy' % num_mix)
    tpr = np.load(cur_tmp_data_path+'parts_tpr_aar_%d.npy' % num_mix)
    plt.plot(tpr,fpr,c=colors[j],marker='o',alpha=.7,lw=0)



plt.show()
        

    

