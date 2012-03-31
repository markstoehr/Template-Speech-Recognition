#
#  Author: Mark Stoehr, 2012
#

#
# main purpose is to test the estimated template on the same
# utterances and see if I get generalized performance
#

import sys, os
sys.path.append('/home/mark/projects/Template-Speech-Recognition')

import template_speech_rec.template_experiments as t_exp
reload(t_exp)
import numpy as np
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.process_scores as ps
from pylab import imshow, plot, figure, show


sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
freq_cutoff = 3000
s_files_path_file = "exp1_path_files_s.txt"
phns_files_path_file = "exp1_path_files_phns.txt"
phn_times_files_path_file = "exp1_path_files_phn_times.txt"
data_dir = "/home/mark/projects/Template-Speech-Recognition/template_speech_rec/Template_Data_Files"
pattern = np.array(('aa','r'))
kernel_length = 7


exp = t_exp.Experiment(sample_rate,freq_cutoff,
                       num_window_samples,
                       num_window_step_samples,
                       fft_length,s_files_path_file,
                       phns_files_path_file,
                       phn_times_files_path_file,
                       data_dir,pattern,kernel_length)


registered_templates=np.load('registered_templates.npy')
mean_template=np.load('mean_template.npy')
template_height ,template_length= mean_template.shape
bgds_mat = np.load('bgds_mat.npy')

mean_background = .3*np.random.rand(template_height) + .1
bg_len = 26
true_pos_thresholds = []
false_pos = []
num_frames = 0
for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    s = exp.get_s(path_idx)
    E,edge_feature_row_breaks,\
        edge_orientations= exp.get_edgemap_no_threshold(s)
        # add the number of frames, this helps us compute the
        # false positive rate for a given unit of time
    num_frames = num_frames + E.shape[1]
    phn_times = exp.get_phn_times(path_idx)
        # get the backgrounds for all detection spots
    P,C=exp.get_detection_scores(E,mean_template,
                                 bg_len, 
                                 mean_background,
                                 edge_feature_row_breaks,
                                 edge_orientations)
    scores = P+C
    # select the object
    pattern_times = exp.get_pattern_times(E,phns,phn_times,s)
    # find the maxima
    scores_maxima = ps.get_maxima(scores,5)
    maxima_idx = np.arange(scores.shape[0])[scores_maxima]
    # see if we pick up the patterns 
    for i in xrange(len(pattern_times)):
        pattern_array =np.empty(scores.shape[0],dtype=bool)
        pattern_array[:]=False
        # consider something a detection if its within a third of the template length around the start of the pattern
        pattern_array[pattern_times[i][0]-template_length/3:pattern_times[i][0]+template_length/3] = True
        pattern_maxima = np.logical_and(scores_maxima,pattern_array)
        if pattern_maxima.any():
            true_pos_thresholds.append( 
                np.max(scores[pattern_maxima]))
        else:
            # this was a false negative
            pattern_thresholds.append(-np.inf)
            # remove the maxima that are contained within the pattern radius
            # at end of for loop only maxima left will be related to false positives
        scores_maxima[pattern_times[i][0]-template_length/3:pattern_times[i][0]+template_length/3] = False
    new_false_pos = scores[scores_maxima].shape[0]
    # check if there were false positives
    if new_false_pos:
        # will do a clustering here
        # clustered_false_pos = ps.cluster_false_positives(new_false_pos)
        # would want to  do an incremental sorting
        # otherwise we just save them to the array
        false_pos.append(list(scores[scores_maxima]))
