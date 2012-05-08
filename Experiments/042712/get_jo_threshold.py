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

