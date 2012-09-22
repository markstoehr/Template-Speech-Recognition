import numpy as np

root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
#root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/092112/'
tmp_data_path = exp_path+'data/'
paper_path = exp_path+'papers/'
scripts_path = exp_path+'scripts/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
import template_speech_rec.template_experiments as t_exp

from collections import defaultdict


s_fnames = [data_path+'Train/'+str(i+1)+'s.npy' for i in xrange(4619)]
flts_fnames = [data_path+'Train/'+str(i+1)+'feature_label_transitions.npy' for i in xrange(4619)]
phns_fnames = [data_path+'Train/'+str(i+1)+'phns.npy' for i in xrange(4619)]

abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])

spread_length=3
abst_threshold=abst_threshold
fft_length=512
num_window_step_samples=80
freq_cutoff=3000
sample_rate=16000
num_window_samples=320
kernel_length=7
offset=3

syllables = np.array([['aa','r'],['p','aa'],['t','aa'],['k','aa'],['b','aa'],['d','aa'],['g','aa']])


out = open(tmp_data_path+'example_E_dict.pkl','rb')
example_E_dict = cPickle.load(out)
out.close()

out = open(tmp_data_path+'example_S_dict.pkl','rb')
example_S_dict = cPickle.load(out)
out.close()

avg_bgd = np.load(tmp_data_path+'avg_E_bgd.npy')

import matplotlib.pyplot as plt



for syllable,examples_S in example_S_dict.items():
   multi_page_display(paper_path+syllable+'_example_S.pdf',
                      examples_S)

