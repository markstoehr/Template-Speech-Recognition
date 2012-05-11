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
    def __init__(self,template,bg,
                 part_length,
                 parts=None,
                 part_starts=None):
        # handle the case where parts are given along with
        # the problem versus the case where they are not given
        if parts:
            self.parts = parts
        else:
            self.parts = np.array([template[:,:part_length],
                          template[:,-part_length:]])
        if part_starts:
            self.part_starts = part_starts
        else:
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


texp = template_exp.\
    Experiment(pattern=np.array(('l','iy')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )




train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)

all_patterns = []
all_pattern_parts = []
E_avg = template_exp.AverageBackground()            
train_data_iter.spread_length = 5

for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_pattern_parts=True,
                            compute_patterns=True,
                            max_template_length=40):
        # the context length is 11
        all_patterns.extend(train_data_iter.patterns)
        all_pattern_parts.extend(train_data_iter.pattern_parts)
        E_avg.add_frames(train_data_iter.E,
                         train_data_iter.edge_feature_row_breaks,
                         train_data_iter.edge_orientations,
                         train_data_iter.abst_threshold)
    else:
        break





output = open('train_patterns_liy051012.pkl','wb')
cPickle.dump(all_patterns,output)
cPickle.dump(all_pattern_parts,output)
output.close()

_,_ ,\
    registered_ex_l,l_template \
    = et.simple_estimate_template([ex[0] for ex in all_pattern_parts])

np.save('registered_ex_l051012',registered_ex_l)
np.save('l_template051012',l_template)

_,_ ,\
    registered_ex_iy,iy_template \
    = et.simple_estimate_template([ex[1] for ex in all_pattern_parts])

np.save('registered_ex_iy051012',registered_ex_iy)
np.save('iy_template051012',iy_template)

_,_ ,\
    registered_ex_liy,liy_template \
    = et.simple_estimate_template(all_patterns)

np.save('registered_ex_liy051012',registered_ex_liy)
np.save('liy_template051012',liy_template)

mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)

np.save('mean_background_liy051012',mean_background)

tpm_liy = TwoPartModel(liy_template,
                       2*template_shape[1]/3,mean_background)



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
            
                                                   

