import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import template_speech_rec.roc_functions as rf
import matplotlib.pyplot as plt
import cPickle,os, pickle,collections

sp = gtrd.SpectrogramParameters(
    sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False)

ep = gtrd.EdgemapParameters(block_length=40,
                            spread_length=1,
                            threshold=.7)


utterances_path = '/var/tmp/stoehr/Template-Speech-Recognition/Data/Train/'
file_indices = gtrd.get_data_files_indices(utterances_path)

phns = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 'sp', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh', 'sil']

phn = ('aa')

# timit mapping
leehon_mapping_base = { 'aa':'aa',
                   'ao':'aa',
                   'ax':'ah',
                   'ax-h':'ah',
                   'axr': 'er',
                   'hv':'hh',
                   'ix':'ih',
                   'el':'l',
                   'em':'m',
                   'en':'n',
                   'nx':'n',
                   'eng':'ng',
                   'zh':'sh',
                   'ux':'uw',
                   'pcl':'sil',
                   'tcl':'sil',
                   'kcl':'sil',
                   'bcl':'sil',
                   'dcl':'sil',
                   'gcl':'sil',
                   'h#':'sil',
                   'pau':'sil',
                   'epi':'sil',
                   'sp':'sil'}


phns_down = list(set([p for p in phns if p not in leehon_mapping_base.keys() and p != 'q'] + [leehon_mapping_base[p] for p in phns if p in leehon_mapping_base.keys() and p != 'q']))

leehon_mapping = {}
for k,v in leehon_mapping_base.items():
    leehon_mapping[k] = v

for phn in phns_down:
    if phn not in leehon_mapping.keys():
        leehon_mapping[phn] = phn

use_phns = np.array(list(set(leehon_mapping.values())))
phn_idx = 0
phn = use_phns[phn_idx]
phn_features,avg_bgd=gtrd.get_syllable_features_directory(utterances_path,file_indices,phn,
                                    S_config=sp,E_config=ep,offset=0,
                                    E_verbose=False,return_avg_bgd=True,
                                                               waveform_offset=15)
