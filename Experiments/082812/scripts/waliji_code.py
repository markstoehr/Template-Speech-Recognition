import numpy as np
root_path = '/var/tmp/stoehr/Template-Speech-Recognition/'
#root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/'
exp_path = root_path + 'Experiments/082812/'
tmp_data_path = exp_path+'data/'
scripts_path = exp_path+'scripts/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.extract_local_features as elf
from fast_patch_coding import fast_patch_coding
import collections

WalijiPart = collections.namedtuple('WalijiPart','log_template log_invtemplate')

def make_WalijiPart_tuple(parts):
    return tuple( WalijiPart(log_template=np.log(part),
                  log_invtemplate=np.log(1-part))
                  for part in parts)




def code_features(E,parts,spec_avg_parts=None,
                  lower_quantile=.9,upper_quantile=1.):
    """
    Part coding for an utterance using the waliji parts
    Optionally will produce the spectrogram associated with the features
    this is important for visualizing and diagnosing what this function is doing
    """
    num_features = E.shape[0]
    num_freq_bands = num_features/8
    edge_feature_row_breaks = np.arange(9,dtype=int)*num_freq_bands
    patch_height, patch_width = parts[0].log_template.shape
    patch_height /= 8
    bps,all_patch_rows,all_patch_cols = elf.extract_local_features_tied(E,patch_height,
                                                                       patch_width, lower_quantile,
                                                                       upper_quantile,
                                                                       edge_feature_row_breaks,
                                                                       segment_ms=500,
                                                                       hop_ms = 5)
    # initially set all features to background
    feature_map = -1*np.ones((num_freq_bands,E.shape[1]),dtype=np.int8)
    # read in all the parts
    patch_ids = np.array(
        tuple(
            np.argmax(
                tuple(
    (bp*part.log_template + (1-bp)*part.log_invtemplate).sum()
    for part in parts)).astype(np.uint16)
    for bp in bps))
    feature_map[all_patch_rows,all_patch_cols] = patch_ids
    if spec_avg_parts is not None:
        S_coded = S_code_features(feature_map,patch_ids,all_patch_rows,all_patch_cols,
                                spec_avg_parts)
        return feature_map,S_coded
    return feature_map

def S_code_features(feature_map,patch_ids,all_patch_rows,all_patch_cols,
                  spec_avg_parts):
    S_coded = np.zeros(feature_map.shape,dtype = np.float32) - np.float32(2*np.abs(spec_avg_parts.min()))
    S_min = S_coded.min()
    S_patch_counter = np.zeros(S_coded.shape,dtype = np.float32)
    fast_patch_coding(spec_avg_parts,
                      S_coded,
                      S_patch_counter,
                      patch_ids,
                      all_patch_rows,
                      all_patch_cols)
    S_true_min = S_coded[S_coded>S_min].min()
    S_coded[S_coded==S_min] = S_true_min
    return S_coded

