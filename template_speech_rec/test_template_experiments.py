import sys, os
sys.path.append('/home/mark/projects/Template-Speech-Recognition')

import template_speech_rec.template_experiments as t_exp
reload(t_exp)
import numpy as np
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
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

# testing template_experiments.py function
# get_detection_scores
E = np.zeros((2,20))
ones_run_length = 5
E[:,7:7+ones_run_length] = np.ones((2,ones_run_length))
template = .95* np.ones((2,ones_run_length))
bgd_length = 2
mean_background = .1* np.ones(2)
edge_feature_row_breaks = np.array([0,1])
edge_orientations = np.array([[0,0]])
P,C=exp.get_detection_scores(E,template,bgd_length, mean_background,
                             edge_feature_row_breaks,
                             edge_orientations)


