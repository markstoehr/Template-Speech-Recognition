#  Testing whether I can detect well on the registered training examples
#    - process utterances
#    - perform signal processing up to thresholding
#    - select out object
#    - register the objects
#    - compute the mean see if we get a performance
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

all_patterns = []
all_bgds = []
bg_len = 26
empty_bgds = []
pattern_num = 0
for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if exp.has_pattern(phns):
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # select the object
        patterns = exp.get_patterns(E,phns,phn_times,s)
        bgds = exp.get_pattern_bgds(E,phns,phn_times,s,bg_len)
        # threshold pattern edges
        for i in xrange(len(patterns)):
            esp.threshold_edgemap(patterns[i],.30,edge_feature_row_breaks)
            esp.spread_edgemap(patterns[i],edge_feature_row_breaks,edge_orientations)
            if bgds[i].shape[1] > 0:
                esp.threshold_edgemap(bgds[i],.30,edge_feature_row_breaks)
                esp.spread_edgemap(bgds[i],edge_feature_row_breaks,edge_orientations)
                # compute background mean
                bgds[i] = np.mean(bgds[i],axis=1)
                # impose floor and ceiling constraints on values
                bgds[i] = np.maximum(np.minimum(bgds[i],.4),.05)
            else:
                bgds[i] = np.random.rand(patterns[i].shape[0]).reshape(patterns[i].shape[0],1)
                bgds[i] = np.mean(bgds[i],axis=1)
                bgds[i] = np.maximum(np.minimum(bgds[i],.4),.05)
                empty_bgds.append(pattern_num)
        pattern_num += len(patterns)
        all_patterns.extend(patterns)
        all_bgds.extend(bgds)

mean_background = np.zeros((all_patterns[0].shape[0],))
empty_idx = 0
num_empties_left = len(empty_bgds)
num_in_avg = 0
for bgd_idx in xrange(len(all_bgds)):
    if empty_idx >= len(empty_bgds):
        num_in_avg +=1
        mean_background = (mean_background * (num_in_avg-1)+all_bgds[bgd_idx])/float(num_in_avg)
    elif bgd_idx == empty_bgds[empty_idx]:
        empty_idx += 1
    else:
        num_in_avg +=1
        mean_background = (mean_background * (num_in_avg-1)+all_bgds[bgd_idx])/float(num_in_avg)

for empty_bgd in xrange(len(empty_bgds)):
    all_bgds[empty_bgds[empty_bgd]] = mean_background.copy()



template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns)

bgds_mat = np.array(all_bgds).transpose()


# collect together the different backgrounds

np.save('template_height',template_height)
np.save('template_length',template_length)
np.save('registered_templates',registered_templates)
np.save('mean_template',mean_template)
np.save('bgds_mat',bgds_mat)

"""
template_height = np.load('template_height.npy')
template_length = np.load('template_length.npy')
registered_templates=np.load('registered_templates.npy')
mean_template=np.load('mean_template.npy')
bgds_mat = np.load('bgds_mat.npy')

"""

'''

'''

# get backgrounds in order to compute tests
template_height = 366;
num_patterns =registered_templates.shape[0]
bgds_mat = np.zeros((template_height,num_patterns))

template_length = 32


P,C = tt.score_template_background_section(mean_template,bgds_mat[:,0],
                                  registered_templates[0])


# get negative examples for testing
neg_bdg_mat = np.zeros((template_height,exp.num_data))
neg_patterns = np.zeros((exp.num_data,template_height,template_length))
cur_bgd = 0
for path_idx in range(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if True:
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # select the object
        neg_pattern,neg_bgd = exp.get_patterns_negative(E,phns,phn_times,s,template_length)
        # threshold pattern edges
        esp.threshold_edgemap(neg_pattern,.30,edge_feature_row_breaks)
        esp.threshold_edgemap(neg_bgd,.30,edge_feature_row_breaks)
        esp.spread_edgemap(neg_pattern,edge_feature_row_breaks,edge_orientations)
        esp.spread_edgemap(neg_bgd,edge_feature_row_breaks,edge_orientations)        
        neg_bgd = np.mean(neg_bgd,axis=1)
        neg_bgd = np.maximum(np.minimum(neg_bgd,.4),.1)
        neg_patterns[path_idx] = neg_pattern
        neg_bdg_mat[:,path_idx] = neg_bgd

# had to quit             
num_neg_patterns = 853

neg_bgd_mat = neg_bdg_mat[:,:num_neg_patterns].copy()
neg_patterns = neg_patterns[:num_neg_patterns].copy()
        
np.save('neg_bgd_mat',neg_bgd_mat)
np.save('neg_patterns',neg_patterns)

        
        
#
# Now we compare scores
#
#    
reg_scores = np.zeros(registered_templates.shape[0])
for t_idx in xrange(registered_templates.shape[0]):
    P,C = tt.score_template_background_section(mean_template,bgds_mat[:,t_idx],
                                               registered_templates[t_idx])
    reg_scores[t_idx] = P+C

#num_neg_patterns = neg_patterns.shape[0]
num_neg_patterns = 853
neg_scores = np.zeros(num_neg_patterns)
for n_idx in xrange(num_neg_patterns):
    P,C = tt.score_template_background_section(mean_template,neg_bgd_mat[:,n_idx],
                                               neg_patterns[n_idx])
    neg_scores[n_idx] = P+C


# create the ROC curves
reg_scores = -reg_scores    
reg_scores.sort()
reg_scores = -reg_scores

neg_scores = -neg_scores
neg_scores.sort()
neg_scores = -neg_scores


# both are now in descending order

neg_pointer = 0
num_neg_for_pos_idx = np.zeros(reg_scores.shape[0])
for reg_idx in xrange(len(reg_scores)):
    while neg_scores[neg_pointer+1] > reg_scores[reg_idx]:
        if neg_pointer < neg_scores.shape[0]-1:
            neg_pointer +=1
    num_neg_for_pos_idx[reg_idx] = neg_pointer/float(neg_scores.shape[0])
    

#
# Test against the library of patterns where we don't do registration
#  want to grab some context around the positive pattern
#
#
all_patterns_context_bgds = []

bg_len = 26
empty_bgds = []
pattern_num = 0
for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if exp.has_pattern(phns):
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # select the object
        # the true indicatesthat we are grabbing context with the pattern
        patterns = exp.get_patterns(E,phns,phn_times,s,True,template_length)
        bgds = exp.get_pattern_bgds(E,phns,phn_times,s,bg_len,True,template_length)
        # threshold pattern edges
        for i in xrange(len(patterns)):
            if bgds[i].shape[1] <= 0:
                bgds[i] = np.random.rand(patterns[i].shape[0]).reshape(patterns[i].shape[0],1)
                empty_bgds.append(pattern_num)
            all_patterns_context_bgds.append((patterns[i],bgds[i]))
        pattern_num += len(patterns)

"""
cPickle.dump(all_patterns_context_bgds,open('all_patterns_context.pickle','w'))
cPickle.dump(empty_bgds,open('all_patterns_context_empty_bgds.pickle','w'))
"""

mean_background = np.zeros((all_patterns[0].shape[0],))
empty_idx = 0
num_empties_left = len(empty_bgds)
num_in_avg = 0
for idx in xrange(len(all_patterns_context_bgds)):
    if empty_idx >= len(empty_bgds):
        num_in_avg +=1
        mean_background = (mean_background * (num_in_avg-1)+all_bgds[bgd_idx])/float(num_in_avg)
    elif bgd_idx == empty_bgds[empty_idx]:
        empty_idx += 1
    else:
        num_in_avg +=1
        mean_background = (mean_background * (num_in_avg-1)+all_bgds[bgd_idx])/float(num_in_avg)

for empty_bgd in xrange(len(empty_bgds)):
    all_bgds[empty_bgds[empty_bgd]] = mean_background.copy()


#
# handle the empty backgrounds later
# going to simply try to run the experiment
#

for idx in xrange(len(all_patterns_context_bgds)):
    if idx not in empty_bgds:
        cur_segment = all_patterns_context_bgds[idx][0]
        cur_bgd = all_patterns_context_bgds[idx][1]
        num_detections = cur_segment.shape[1]-template_length+1
        detect_array = np.zeros(num_detections)
        for d in xrange(num_detections):
            test_segment = cur_segment[]


#
# Going to run the thing on whole utterances
#
#

mean_background = .3*np.random.rand(template_height) + .1
bg_len = 26
true_pos_thresholds = []
false_pos = []
num_frames = 0
for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if exp.has_pattern(phns):
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # get the backgrounds for all detection spots
        P,C=exp.get_detection_scores(E,mean_template,
                                     bgd_length, 
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
                true_pos_thresholds.append( np.max(scores[pattern_maxima]))
            else:
                # this was a false negative
                pattern_thresholds.append(-np.inf)
            # remove the maxima that are contained within the pattern radius
            # at end of for loop only maxima left will be related to false positives
            scores_maxima[patterns[i][0]:patterns[i][1]] = False
        new_false_pos = scores[scores_maxima].shape[0]
        # check if there were false positives
        if new_false_pos:
            # will do a clustering here
            # clustered_false_pos = ps.cluster_false_positives(new_false_pos)
            # would want to  do an incremental sorting
            # otherwise we just save them to the array
            false_pos.append(list(new_false_pos))
            
            
            
