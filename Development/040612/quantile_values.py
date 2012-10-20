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
from scipy.stats import histogram
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab


sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
freq_cutoff = 3000
s_files_path_file = "/home/mark/projects/Template-Speech-Recognition/Experiments/032211/exp1_path_files_s.txt"
phns_files_path_file = "/home/mark/projects/Template-Speech-Recognition/Experiments/032211/exp1_path_files_phns.txt"
phn_times_files_path_file = "/home/mark/projects/Template-Speech-Recognition/Experiments/032211/exp1_path_files_phn_times.txt"
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



#
# First experiment script is to simply get all the values
# for the order stats when simply considering the patterns
#


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
        pattern_num += len(patterns)
        all_patterns.extend(patterns)
        all_bgds.extend(bgds)

# now compute the pattern values
tau_alpha_vals = np.empty((len(all_patterns),8))


for i in xrange(len(all_patterns)):
    a,tau_alpha_vals[i] = esp.threshold_edgemap(all_patterns[i],.30,edge_feature_row_breaks,report_level=True)
    

# looking at histograms from data
fig = plt.figure()
ax = fig.add_subplot(111)
n, bins, patches = ax.hist(np.log(tau_alpha_vals[:,0]), 20, normed=1, facecolor='green', alpha=0.75)
bincenters = 0.5*(bins[1:]+bins[:-1])
ax.grid(True)
plt.show()

p = all_patterns[i].copy()




#
# Going to do template averaging and then see which values
# of alpha give the closest match to the template
# or see which values for the edge thresholding give the best values, so, namely need another parameter

template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns)


#
#  Get quantiles for the alpha value .1 .2 .3 .4 .5
#

quantiles = np.array([.1,.2,.3,.4,.5])
quantile_levels = np.empty((quantiles.shape[0],
                            tau_alpha_vals.shape[1]))
for edge_type in xrange(quantile_levels.shape[1]):
    taus = tau_alpha_vals[:,edge_type].copy()
    taus.sort()
    quantile_levels[:,edge_type] = map(lambda i: taus[i],
                                       taus.shape[0]*quantiles)

# we should also compute the quantiles for the background thresholding and see about those
    
bgd_tau_alpha_vals =  np.empty((len(all_bgds),8))

for i in xrange(len(all_bgds)):
    a,bgd_tau_alpha_vals[i] = esp.threshold_edgemap(all_bgds[i],.30,edge_feature_row_breaks,report_level=True)


quantiles = np.array([.1,.2,.3,.4,.5])
bgd_quantile_levels = np.empty((quantiles.shape[0],
                            tau_alpha_vals.shape[1]))
for edge_type in xrange(quantile_levels.shape[1]):
    taus = bgd_tau_alpha_vals[:,edge_type].copy()
    taus.sort()
    bgd_quantile_levels[:,edge_type] = map(lambda i: taus[i],
                                       taus.shape[0]*quantiles)


utt_tau_alpha_vals =  np.empty((exp.num_data,8))


# now we look at the utterance level quantile levels
for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    # check if this datum has what we need
    s = exp.get_s(path_idx)
    E,edge_feature_row_breaks,\
        edge_orientations= exp.get_edgemap_no_threshold(s)
    a,utt_tau_alpha_vals[path_idx] = esp.threshold_edgemap(E,.30,edge_feature_row_breaks,report_level=True)

path_idx
utt_tau_alpha_vals_pref = utt_tau_alpha_vals[:214]
    
#
# we want to compare the values
# important to think about the consequences of the 
# comparison

    
mean_template_match = np.empty((quantile_level.shape[0],
                                len(all_patterns)))

for i in xrange(len(all_patterns)):
    # get the background signal processing done
    if all_bgds[i].shape[1] > 0:
        
    # process the edges for each of the different thresholds
    for j in xrange(quantile_levels.shape[0]):
        a,tau_alpha_vals[i] = esp.threshold_edgemap(all_patterns[i],.30,edge_feature_row_breaks, 
                                                    abst_threshold=)
    esp.spread_edgemap(a,edge_feature_row_breaks,edge_orientations)



for i in xrange(len(all_patterns)):
    a,tau_alpha_vals[i] = esp.threshold_edgemap(p,.30,edge_feature_row_breaks,report_level=True)

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
