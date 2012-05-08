#
# Going to see what template thresholds do best with the mean template
#
#


import sys, os
sys.path.append('/home/mark/projects/Template-Speech-Recognition')
#sys.path.append('/home/mark/projects/Template-Speech-Recognition')

import template_speech_rec.template_experiments as t_exp
reload(t_exp)
import numpy as np
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.process_scores as ps
from pylab import imshow, plot, figure, show
import template_speech_rec.bernoulli_em as bem
import template_speech_rec.parts_model as pm
import cPickle


sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
freq_cutoff = 3000
exp_path_files_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/032211/'
s_files_path_file = exp_path_files_dir+"exp1_path_files_s.txt"
phns_files_path_file = exp_path_files_dir+"exp1_path_files_phns.txt"
phn_times_files_path_file = exp_path_files_dir+"exp1_path_files_phn_times.txt"

# /home/mark/projects
data_dir = "/home/mark/projects/Template-Speech-Recognition/template_speech_rec/Template_Data_Files"
#data_dir = "/"
#data_dir = "/var/tmp/stoehr/Projects/edges/WavFilesTrain/"
pattern = np.array(('aa','r'))
kernel_length = 7


exp = t_exp.Experiment(sample_rate,freq_cutoff,
                       num_window_samples,
                       num_window_step_samples,
                       fft_length,s_files_path_file,
                       phns_files_path_file,
                       phn_times_files_path_file,
                       data_dir,pattern,kernel_length)

#
# Get the templates
#
#

class AverageBackground:
    def __init__(self):
        self.num_frames = 0
        self.processed_frames = False
    # Method to add frames
    def add_frames(self,E,edge_feature_row_breaks,
                   edge_orientations,abst_threshold):
        new_E = E.copy()
        esp.threshold_edgemap(new_E,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(new_E,edge_feature_row_breaks,edge_orientations,spread_length=3)
        if not self.processed_frames:
            self.E = np.mean(new_E,axis=1)
            self.processed_frames = True
        else:
            self.E = (self.E * self.num_frames + np.sum(new_E,axis=1))/(self.num_frames+new_E.shape[1])
        self.num_frames += new_E.shape[1]
        

E_avg = AverageBackground()            


all_patterns = []
# these are the fronts and backs of the syllable saved to estimate the template bits
# just randomly these are going to be length 10 with five of the frames from before the syllable start
all_fronts_backs = []
all_bgds = []
bg_len = 26
empty_bgds = []
pattern_num = 0
abst_threshold = np.array([0.025,0.025,0.015,0.015,
                           0.02,0.02,0.02,0.02])

all_raw_patterns_context = []
all_raw_bgds = []


# number of thresholds per data point is 8
pattern_edge_thresholds = []
pattern_lengths = []
bgd_edge_thresholds=[]


mean_template_length = 33

for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if True: #exp.has_pattern(phns):
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # select the object
        patterns = exp.get_patterns(E,phns,phn_times,s)
        patterns_context = exp.get_patterns(E,phns,phn_times,s,context=True,template_length=33)
        bgds = exp.get_pattern_bgds(E,phns,phn_times,s,bg_len)
        fronts_backs = exp.get_pattern_fronts_backs(E,phns,phn_times,s,bg_len)
        E_avg.add_frames(E,edge_feature_row_breaks,
                   edge_orientations,abst_threshold)
        # threshold pattern edges
        for i in xrange(len(patterns)):
            all_raw_patterns_context.append(patterns_context[i].copy())
            all_raw_bgds.append(bgds[i].copy())
            _, edge_thresholds = esp.threshold_edgemap(patterns[i],.30,edge_feature_row_breaks,report_level=True,abst_threshold=abst_threshold)
            # we record both the thresholds
            # and the length to see if there is a relationship
            pattern_edge_thresholds.append(edge_thresholds)
            pattern_lengths.append(patterns[i].shape[1])
            esp.spread_edgemap(patterns[i],edge_feature_row_breaks,edge_orientations,spread_length=5)
            if bgds[i].shape[1] > 0:
                _,edge_thresholds = esp.threshold_edgemap(bgds[i],.30,edge_feature_row_breaks,report_level=True,abst_threshold=abst_threshold)
                bgd_edge_thresholds.append(edge_thresholds)
                esp.spread_edgemap(bgds[i],edge_feature_row_breaks,edge_orientations,spread_length=5)
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


all_raw_patterns_context_array = np.empty(len(all_raw_patterns_context),dtype=np.ndarray)

for i in range(len(all_raw_patterns_context)):
    all_raw_patterns_context_array[i] = all_raw_patterns_context[i]

all_raw_patterns_context_array


edge_thresholds_array = np.empty(len(edge_thresholds),
                                 dtype=np.ndarray)
for i in range(len(edge_thresholds)):
    edge_thresholds_array[i] = edge_thresholds[i]


np.save('all_raw_patterns_context042812',all_raw_patterns_context_array)
np.save('edge_thresholds042812',edge_thresholds_array)


mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)


template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns,template_length=33)


# pattern_edge_thresholds
pattern_edge_thresholds = np.array(pattern_edge_thresholds)
num_edges = 8

# construct a kernel density estimator

from scipy import (stats,mgrid, c_,
reshape, rot90, random)

from pylab import (xlabel, ylabel,
cm,close)


for i in range(num_edges):
    edge_threshold_sorted = np.sort(pattern_edge_thresholds[:,i])
    # get 4 neighbor diff
    bwidth = np.mean(-(edge_threshold_sorted[4:] - edge_threshold_sorted[:-4])**2)/np.log(.01)
    kernel = stats.kde.gaussian_kde(edge_threshold_sorted)
    lo = edge_threshold_sorted[0]
    hi = edge_threshold_sorted[-1]
    positions = np.arange(lo,hi, (hi-lo)/100.)
    Z = kernel(positions)
    plot(positions,Z)
    show()
    print "The threshold is at",abst_threshold[i]
    raw_input("press enter to continue")
    close()
    

#
# see what the maximum response is for all of these
# find what the edge values that give the highest scores
# on all the patterns

abst_threshold_check_range = np.arange(.001,.04,.005)

#all_raw_patterns_context_array
thresholds_score = np.empty((all_raw_patterns_context_array.shape[0],num_edges,abst_threshold_check_range.shape[0]))
bgd = mean_background


# try to build up the best abst threshold vector
cur_abst = abst_threshold.copy()
spread_length=5
for edge in xrange(num_edges):
    for thresh in xrange(abst_threshold_check_range.shape[0]):
        print "Considering edge",edge," and threshold",abst_threshold_check_range[thresh]
        cur_abst[edge] = abst_threshold_check_range[thresh]
        for pattern_id in xrange(all_raw_patterns_context_array.shape[0]):
            cur_section = all_raw_patterns_context_array[pattern_id].copy()
            scores = np.empty(cur_section.shape[1] - template_length)
            for t in xrange(cur_section.shape[1] - template_length):
                E_segment = cur_section[:,t:t+template_length].copy()
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=cur_abst)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
                P,C = tt.score_template_background_section(mean_template,bgd,E_segment)
                scores[t] = P+C
            thresholds_score[pattern_id,edge,thresh] = np.max(scores)
        # best peformance for the choice of edge type
        mean_scores_edge = np.mean(thresholds_score,axis=0)[edge]
        best_thresh = np.argmax(mean_scores_edge)
        cur_abst[edge] = abst_threshold_check_range[best_thresh]
                
"""
>>> print cur_abst
[ 0.001  0.001  0.001  0.001  0.001  0.001  0.001  0.001]
"""
#
# very interesting, it appears that we might be better off with the edge
#    thresholds much lower


abst_threshold_check_range = np.arange(-.02,.001,.005)

#all_raw_patterns_context_array
thresholds_score = np.empty((all_raw_patterns_context_array.shape[0],num_edges,abst_threshold_check_range.shape[0]))
bgd = mean_background


# try to build up the best abst threshold vector
cur_abst = abst_threshold.copy()
spread_length=5
for edge in xrange(num_edges):
    for thresh in xrange(abst_threshold_check_range.shape[0]):
        print "Considering edge",edge," and threshold",abst_threshold_check_range[thresh]
        cur_abst[edge] = abst_threshold_check_range[thresh]
        for pattern_id in xrange(all_raw_patterns_context_array.shape[0]):
            cur_section = all_raw_patterns_context_array[pattern_id].copy()
            scores = np.empty(cur_section.shape[1] - template_length)
            for t in xrange(cur_section.shape[1] - template_length):
                E_segment = cur_section[:,t:t+template_length].copy()
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=cur_abst)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
                P,C = tt.score_template_background_section(mean_template,bgd,E_segment)
                scores[t] = P+C
            thresholds_score[pattern_id,edge,thresh] = np.max(scores)
        # best peformance for the choice of edge type
        mean_scores_edge = np.mean(thresholds_score,axis=0)[edge]
        best_thresh = np.argmax(mean_scores_edge)
        cur_abst[edge] = abst_threshold_check_range[best_thresh]


    
    
#
#
# we also want to test what thresholding gives the best score for patterns
#
#

#
#
# another comparison is to look at the false detections and
# to see how well we do on those levels as we change the
# threshold


score_vals_edge_thresholds = []




for path_idx in xrange(exp.num_data):
    print "on path", path_idx
    phns = exp.get_phns(path_idx)
    #if not exp.has_pattern(phns):
    #    continue
    phn_times = exp.get_phn_times(path_idx)
    s = exp.get_s(path_idx)
    E,edge_feature_row_breaks,\
        edge_orientations= exp.get_edgemap_no_threshold(s)
    feature_start, \
        feature_step, num_features =\
        esp._get_feature_label_times(s,
                                         exp.num_window_samples,
                                         exp.num_window_step_samples)
    feature_labels, \
        feature_label_transitions \
        = esp._get_labels(phn_times,
                      phns,
                      feature_start, feature_step, 
                      num_features,
                      exp.sample_rate)
        # add the number of frames, this helps us compute the
        # false positive rate for a given unit of time
    num_frames = num_frames + E.shape[1]
    phn_times = exp.get_phn_times(path_idx)
    pattern_times = exp.get_pattern_times(phns,phn_times,s)
    for t in xrange(E.shape[1] - template_length):
        do_continue = False
        for pattern_id in pattern_times:
            if pattern_id[0]-template_length/3 < t \
                    or pattern_id[0]+template_length/3 > t:
                do_continue = True
        if do_continue: continue
        if t < bg_len:
            cur_bgd = mean_background
        else:
            cur_bgd = np.maximum(np.minimum(np.mean(E[:,t-bg_len:t],
                                                    axis=1),
                                            .4),
                                 .05)
        E_segment = E[:,t:t+template_length].copy()
        _,edge_thresholds = esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
        P,C = tt.score_template(template,cur_bgd,E_segment)
        score_vals_edge_thresholds.append((P+C,edge_thresholds))
    patterns = exp.get_patterns(E,phns,phn_times,s)
    bgds = exp.get_pattern_bgds(E,phns,phn_times,s,bg_len)
    # find the maxima
    scores_maxima = ps.get_maxima(scores,maxima_radius)
    maxima_idx = np.arange(scores.shape[0])[scores_maxima]
    # see if we pick up the patterns 
    for i in xrange(len(pattern_times)):
        esp.threshold_edgemap(patterns[i],.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(patterns[i],edge_feature_row_breaks,edge_orientations,spread_length=3)
        esp.threshold_edgemap(bgds[i],.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(bgds[i],edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
        # compute background mean
        bgds[i] = np.mean(bgds[i],axis=1)
        # impose floor and ceiling constraints on values
        bgds[i] = np.maximum(np.minimum(bgds[i],.4),.05)
        registered_example = np.empty((template_height,template_length))
        et._register_template(patterns[i],registered_example,template_height,template_length)
        P_reg,C_reg = tt.score_template_background_section(mean_template,bgds[i],registered_example)        
        pattern_array =np.empty(scores.shape[0],dtype=bool)
        pattern_array[:]=False
        # consider something a detection if its within a third of the template length around the start of the pattern
        pattern_array[pattern_times[i][0]-int(np.ceil(template_length/3)):pattern_times[i][0]+int(np.ceil(2*template_length/5.))] = True
        pattern_maxima = np.logical_and(scores_maxima,pattern_array)
        if pattern_maxima.any():
            max_true_threshold = np.max(scores[pattern_maxima])
            true_pos_thresholds.append( 
                (max_true_threshold,np.arange(scores.shape[0])[scores == max_true_threshold][0],P_reg+C_reg,path_idx))
        else:
            # this was a false negative
            true_pos_thresholds.append((-np.inf,P_reg+C_reg,path_idx))
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
        # have a tuple that gives the scores their locations and the path id
        false_pos.append((scores[scores_maxima],np.arange(scores.shape[0])[scores_maxima],
                          np.max(scores[scores_maxima]),
                          path_idx))


