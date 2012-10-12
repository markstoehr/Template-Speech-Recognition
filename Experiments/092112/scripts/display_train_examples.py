import numpy as np

#root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
root_path = '/home/mark/Template-Speech-Recognition/'
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
avg_bgd_S = -10 * np.ones(example_S_dict[('aa','r')][0].shape[0])

max_lengths = dict( (k,max(e.shape[1] for e in v)) for k,v in example_S_dict.items())

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def pad_example_bgd(S,bgd,max_length):
   if S.shape[1] < max_length:
      return np.hstack((S,np.tile(bgd,(max_length-S.shape[1],1)).T))
   else:
      return S

def multi_page_display(f_path, examples_list,avg_bgd,aspect=3):
   pdf = PdfPages(f_path)
   max_length = max(e.shape[1] for e in examples_list)
   for example in examples_list:
      plt.figure()
      S = pad_example_bgd(example,avg_bgd,max_length)
      plt.imshow(S[::-1],interpolation='nearest',aspect=aspect)
      plt.xlabel('time (ms)')
      plt.ylabel('freq (Hz)')
      pdf.savefig()
      plt.close()
   pdf.close()

multi_page_display('test.pdf',example_S_dict[('aa','r')][:10],avg_bgd_S,aspect=5)

for k,v in example_S_dict.items():
   f_path = paper_path+k[0]+'_'+k[1]+'_examples_S.pdf'
   multi_page_display(f_path,v,avg_bgd_S,aspect=.5)

null_bgd_E = np.ones(example_E_dict[('aa','r')][0].shape[0],dtype=np.uint8)

for k,v in example_E_dict.items():
   print k
   f_path = paper_path+k[0]+'_'+k[1]+'_examples_E.pdf'
   multi_page_display(f_path,v,null_bgd_E,aspect=.5)




f_path = 'test.pdf'
pdf = PdfPages(f_path)

for syllable,examples_S in example_S_dict.items():
   multi_page_display(paper_path+syllable+'_example_S.pdf',
                      examples_S)

