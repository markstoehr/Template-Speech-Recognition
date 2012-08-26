import numpy as np
from collections import defaultdict
import collections
#root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
tmp_data_path = root_path+'Experiments/080612/data/'
old_exp_path = root_path+'Experiments/080312/'
model_save_dir = root_path+'Experiments/080612/models2/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture

phns = ['p','t','k','b','d','g','iy','aa','uw']
num_mixtures_set = [6,10,14]
num_folds = 4




out = open(model_save_dir+"bmodels_parts_padded080812.pkl","rb")
bmodels = cPickle.load(out)
out.close()

apply_bms_to_examples(
                    phn_examples3[dev_mask],
                    phn_lengths[dev_mask],phn_bgs[dev_mask],bmodels[classifier_phn_id][mix_id][fold_num])

import matplotlib.pyplot as plt




out = open(model_save_dir + 'results_for_all_phns_padded_actual_parts081112.pkl','rb')
results_for_all_phns_parts = cPickle.load(out)
out.close()
