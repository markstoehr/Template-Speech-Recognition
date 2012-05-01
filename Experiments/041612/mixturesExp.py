import sys, os
sys.path.append('/var/tmp/stoehr/Projects/Template-Speech-Recognition')
#sys.path.append('/home/mark/projects/Template-Speech-Recognition')

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
exp_path_files_dir = '/var/tmp/stoehr/Projects/hacky_move/Template-Speech-Recognition/Experiments/032211/'
s_files_path_file = exp_path_files_dir+"exp2_path_files_s.txt"
phns_files_path_file = exp_path_files_dir+"exp2_path_files_phns.txt"
phn_times_files_path_file = exp_path_files_dir+"exp2_path_files_phn_times.txt"
data_dir = "/var/tmp/stoehr/Projects/hacky_move/Template-Speech-Recognition/Experiments/032211/Data/Estimate2/"
freq_cutoff = 3000
# exp_path_files_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/032211/'
# s_files_path_file = exp_path_files_dir+"exp1_path_files_s.txt"
# phns_files_path_file = exp_path_files_dir+"exp1_path_files_phns.txt"
# phn_times_files_path_file = exp_path_files_dir+"exp1_path_files_phn_times.txt"
# # /home/mark/projects
# data_dir = "/home/mark/projects/Template-Speech-Recognition/template_speech_rec/Template_Data_Files"
#data_dir = "/home/mark/projects/Template-Speech-Recognition/template_speech_rec/Template_Data_Files"
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
abst_threshold = np.array([0.025,0.025,0.015,0.015,
                           0.02,0.02,0.02,0.02])
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
        E_avg.add_frames(E,edge_feature_row_breaks,
                   edge_orientations,abst_threshold)
        # threshold pattern edges
        for i in xrange(len(patterns)):
            esp.threshold_edgemap(patterns[i],.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
            esp.spread_edgemap(patterns[i],edge_feature_row_breaks,edge_orientations,spread_length=5)
            if bgds[i].shape[1] > 0:
                esp.threshold_edgemap(bgds[i],.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
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


mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)


# use the length from the other examples
template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns,template_length=32)

bgds_mat = np.array(all_bgds).transpose()

template_shape = np.array((template_height,template_length))
np.save('template_shape041612',template_shape)
np.save('registered_templatesExp2041612',registered_templates)
np.save('mean_template041612',mean_template)
np.save('mean_background041612',mean_background)

reg_templatesExp1 = np.load('../RedoAlexey/registered_templates041312.npy')

reg_templates = np.vstack((registered_templates,reg_templatesExp1))

np.save('combinedRegisteredSpecs041612',reg_templates)

"""
reg_templates = np.load('combinedRegisteredSpecs041612.npy')
"""

#
# Now running the EM algorithm, going to actually have to change the detection
# code to take into account the mixture things

import template_speech_rec.bernoulli_em as bem
import random

bm2 = bem.Bernoulli_Mixture(2,reg_templates)
bm2.run_EM(.00001)

####
#
# Going to figure out how the detection is going to work in this case
#

path_idx = 0
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
# get the backgrounds for all detection spots
P,C=exp.get_detection_scores_slow(E,mean_template,
                                 bg_len, 
                                 mean_bgd,
                                 edge_feature_row_breaks,
                                 edge_orientations,
                                 abst_threshold=abst_threshold,
                                 spread_length=3)





mix_scores = []
for mix_id in range(bm.num_mix):
    template_tmp = bm.templates[mix_id]
    mix_Ps,mix_Cs = exp.get_detection_scores_slow(E,template_tmp,bg_len,mean_background,
                                                   edge_feature_row_breaks,
                                                   edge_orientations,
                                                   abst_threshold=abst_threshold,
                                                   spread_length=3)
    mix_scores.append( mix_Ps+mix_Cs + np.log( bm2.weights[mix_id]))

mix_scores = np.array(mix_scores)
5A

    return np.amax(mix_scores,axis=0)


#
#
# now onto testing
#
#

# need get_detection_scores_mixture

sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
s_files_path_file = "test_s_paths.txt"
phns_files_path_file = "test_phns_paths.txt"
phn_times_files_path_file = "test_phn_times_paths.txt"
data_dir = "/"
freq_cutoff = 3000
# exp_path_files_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/032211/'
# s_files_path_file = exp_path_files_dir+"exp1_path_files_s.txt"
# phns_files_path_file = exp_path_files_dir+"exp1_path_files_phns.txt"
# phn_times_files_path_file = exp_path_files_dir+"exp1_path_files_phn_times.txt"
# # /home/mark/projects
# data_dir = "/home/mark/projects/Template-Speech-Recognition/template_speech_rec/Template_Data_Files"
#data_dir = "/home/mark/projects/Template-Speech-Recognition/template_speech_rec/Template_Data_Files"
pattern = np.array(('aa','r'))
kernel_length = 7


exp = t_exp.Experiment(sample_rate,freq_cutoff,
                       num_window_samples,
                       num_window_step_samples,
                       fft_length,s_files_path_file,
                       phns_files_path_file,
                       phn_times_files_path_file,
                       data_dir,pattern,kernel_length)

template_height = 366;
num_patterns =registered_templates.shape[0]


template_length = 32

true_pos_thresholds = []
false_pos = []
num_frames = 0
maxima_radius = 4

spread_length = 3
all_scores = np.empty(exp.num_data,dtype = object)
mean_bgd = mean_background

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
        # get the backgrounds for all detection spots
    scores=exp.get_detection_scores_mix(E,bm2,
                                 bg_len, 
                                 mean_bgd,
                                 edge_feature_row_breaks,
                                 edge_orientations,
                                 abst_threshold=abst_threshold,
                                 spread_length=3)
    print "Computed Scores"
    all_scores[path_idx] = scores.copy()
    # select the object
    pattern_times = exp.get_pattern_times(E,phns,phn_times,s)
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
        # get the backgrounds for all detection spots
    P,C=exp.get_detection_scores_slow(E,mean_template,
                                 bg_len, 
                                 mean_bgd,
                                 edge_feature_row_breaks,
                                 edge_orientations,
                                 abst_threshold=abst_threshold,
                                 spread_length=3)
    scores=P+C
    print "Computed Scores"
    all_scores[path_idx] = scores.copy()
    # select the object
    pattern_times = exp.get_pattern_times(E,phns,phn_times,s)
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



#
# compute the scores with a different maxima neighborhood size
#
#


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
        # get the backgrounds for all detection spots
    P,C=exp.get_detection_scores_slow(E,mean_template,
                                 bg_len, 
                                 mean_bgd,
                                 edge_feature_row_breaks,
                                 edge_orientations,
                                 abst_threshold=abst_threshold,
                                 spread_length=3)
    scores=all_scores[path_idx].copy()
    print "Computed Scores"
    
    # select the object
    pattern_times = exp.get_pattern_times(E,phns,phn_times,s)
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

