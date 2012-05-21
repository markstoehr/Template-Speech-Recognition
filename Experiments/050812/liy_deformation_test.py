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
        self.base_template = template
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
        self.get_min_max_def()
        self.bg = bg
        self.template_height = self.bg.shape[0]
        self.num_templates = self.min_max_def[1] - self.min_max_def[0]
        self.get_def_templates()
        assert(self.parts[0].shape == self.parts[1].shape)
        #
    def get_def_templates(self):
        self.def_templates = np.zeros((self.num_templates,
                                     self.template_height,
                                     self.length_range[1]))
        for t_id in xrange(self.num_templates):
            self.def_templates[t_id] = self.get_def_template(self.min_max_def[0] + t_id)
    #                                 
    def get_def_template(self,def_size):
        def_template = np.tile(self.bg,(self.length_range[1],1)).T
        self.cur_part_starts = self.part_starts.copy()
        self.cur_part_starts[1] += def_size
        def_template[:,self.cur_part_starts[0]:self.cur_part_starts[1]] = self.parts[0][:,:self.cur_part_starts[1]]
        # handle the overlap
        if self.cur_part_starts[1] < self.part_length:
            def_template[:,self.cur_part_starts[1]:self.part_length] = \
                .5 * self.parts[0][:,self.cur_part_starts[1]:] +\
                .5 * self.parts[1][:,:self.part_length-self.cur_part_starts[1]]
        def_template[:,self.part_length:self.part_length+self.cur_part_starts[1]] \
            = self.parts[1][:,self.part_length-self.cur_part_starts[1]:]
        return def_template
        #
    def get_length_range(self):
        self.length_range = np.array((min([p.shape[1] for p in self.parts]),sum([p.shape[1] for p in self.parts])))
    def get_min_max_def(self):
        self.min_max_def = (-self.part_starts[1],self.parts[0].shape[1] - self.part_starts[1])
        return self.min_max_def


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

train_data_iter.reset_exp()
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


template_shape = liy_template.shape

tpm_liy = TwoPartModel(liy_template,mean_background,
                       2*template_shape[1]/3,)
tpm.get_min_max_def


edge_feature_row_breaks = np.load('edge_feature_row_breaks.npy')

edge_orientations = np.load('edge_orientations.npy')
abst_threshold = np.load('abst_threshold.npy')
#mean_template = np.load('mean_template_piy051012.npy')
#mean_background = np.load('mean_background_piy051012.npy')


#tpm = TwoPartModel(mean_template,2*template_shape[1]/3,mean_background)

def_range = tpm_liy.get_min_max_def()

tpm_liy.get_def_template(0)

all_def_templates = np.empty((def_range[1]-def_range[0],
                            tpm_liy.def_template.shape[0],
                            tpm_liy.def_template.shape[1]))
for d in xrange(len(def_range)):
    tpm_liy.get_def_template(def_range[d])
    all_def_templates[d] = tpm_liy.def_template.copy()
    
tpm_liy.get_length_range()

pattern_contexts= []

for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                            compute_patterns_context=True,
                            max_template_length=40):
        # the context length is 11
        pattern_contexts.extend(tune_data_iter.patterns_context)
    else:
        break



optimal_detection_scores = -np.inf * np.ones((len(pattern_contexts),def_range[1]-def_range[0]))
optimal_detection_idx = np.zeros((len(pattern_contexts),def_range[1]-def_range[0]))
for c_id in xrange(len(pattern_contexts)):
    print c_id
    cur_context = pattern_contexts[c_id]
    num_detections = cur_context.shape[1] - tpm_liy.length_range[1]
    win_length = tpm_liy.length_range[1]
    for d in xrange(num_detections):
        E_window = cur_context[:,d:d+win_length].copy()
        esp.threshold_edgemap(E_window,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_window,edge_feature_row_breaks,edge_orientations,spread_length=3)
        # base detection
        for deformation in xrange(def_range[0],def_range[1]):
            def_template = all_def_templates[deformation-def_range[0]]
            P,C = tt.score_template_background_section(def_template,tpm_liy.bg,E_window)
            score = P+C
            if score > optimal_detection_scores[c_id,deformation-def_range[0]]:
                optimal_detection_scores[c_id,deformation-def_range[0]] = score
                optimal_detection_idx[c_id,deformation-def_range[0]] = d
            
np.save('optimal_detection_scores_liy051112',optimal_detection_scores)
np.save('optimal_detection_idx_liy051112',optimal_detection_idx)
np.save('def_range_liy051112',np.array(def_range))

# get the pattern lengths
# see if the mixture model is able to pick up on the noise
# also see how well template does on each of the patterns being considered
# register the examples and see if the examples that are hits for the
# registered examples is reflective for which are the hits if
# the examples are not registered

tune_data_iter.reset_exp()
liy_tune_patterns = []

for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                            compute_patterns=True,
                            max_template_length=40):
        liy_tune_patterns.extend(tune_data_iter.patterns)
    else:
        break

# we want to relate information about the detection scores
# to statistics collected about the 
# liy_tune_patterns, so we need to make sure the lengths
# are right

assert len(liy_tune_patterns) == optimal_detection_scores.shape[0]

output = open('liy_tune_patterns051112','wb')
cPickle.dump(liy_tune_patterns,output)
output.close()

