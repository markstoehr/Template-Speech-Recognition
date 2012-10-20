root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)

#import template_speech_rec.data_loading as dl
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt


class TwoPartModel:
    def __init__(self,template,part_length,bg):
        self.parts = np.array([template[:,:part_length],
                          template[:,-part_length:]])
        self.part_starts = np.array([0,
                                template.shape[1] - part_length])
        self.part_length = part_length
        self.get_length_range()
        self.bg = bg
        assert(self.parts[0].shape == self.parts[1].shape)
        #
    def get_def_template(self,def_size):
        self.def_template = np.tile(self.bg,(self.length_range[1],1)).T
        self.cur_part_starts = self.part_starts.copy()
        self.cur_part_starts[1] += def_size
        self.def_template[:,self.cur_part_starts[0]:self.cur_part_starts[1]] = self.parts[0][:,:self.cur_part_starts[1]]
        # handle the overlap
        if self.cur_part_starts[1] < self.part_length:
            self.def_template[:,self.cur_part_starts[1]:self.part_length] = \
                .5 * self.parts[0][:,self.cur_part_starts[1]:] +\
                .5 * self.parts[1][:,:self.part_length-self.cur_part_starts[1]]
        self.def_template[:,self.part_length:self.part_length+self.cur_part_starts[1]] \
            = self.parts[1][:,self.part_length-self.cur_part_starts[1]:]
        #
    def get_length_range(self):
        self.length_range = np.array((min([p.shape[1] for p in self.parts]),sum([p.shape[1] for p in self.parts])))
    def get_min_max_def(self):
        pass


output = open('tuning_pattern051012.pkl','rb')
tuning_patterns_context = cPickle.load(output)
tuning_patterns_times = cPickle.load(output)
output.close()

edge_feature_row_breaks = np.load('edge_feature_row_breaks.npy')

edge_orientations = np.load('edge_orientations.npy')
abst_threshold = np.load('abst_threshold.npy')
mean_template = np.load('mean_template_piy051012.npy')
mean_background = np.load('mean_background_piy051012.npy')

template_shape = mean_template.shape

tpm = TwoPartModel(mean_template,2*template_shape[1]/3,mean_background)

def_range = np.arange(-10,9)

tpm.get_def_template(0)

all_def_templates = np.empty((def_range.shape[0],
                            tpm.def_template.shape[0],
                            tpm.def_template.shape[1]))
for d in xrange(def_range.shape[0]):
    tpm.get_def_template(def_range[d])
    all_def_templates[d] = tpm.def_template.copy()
    

optimal_detection_scores = -np.inf * np.ones((len(tuning_patterns_context),def_range.shape[0]))
optimal_detection_idx = np.zeros((len(tuning_patterns_context),def_range.shape[0]))
for c_id in xrange(len(tuning_patterns_context)):
    print c_id
    cur_context = tuning_patterns_context[c_id]
    num_detections = cur_context.shape[1] - tpm.length_range[1]
    win_length = tpm.length_range[1]
    for d in xrange(num_detections):
        E_window = cur_context[:,d:d+win_length].copy()
        esp.threshold_edgemap(E_window,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_window,edge_feature_row_breaks,edge_orientations,spread_length=3)
        # base detection
        for deformation in def_range:
            def_template = all_def_templates[10+deformation]
            P,C = tt.score_template_background_section(def_template,tpm.bg,E_window)
            score = P+C
            if score > optimal_detection_scores[c_id,deformation]:
                optimal_detection_scores[c_id,deformation] = score
                optimal_detection_idx[c_id,deformation] = d
            
                                                   

