root_path = '/home/mark/projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)

#import template_speech_rec.data_loading as dl
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt


output = open('all_patterns_piy050912.pkl','rb')
all_patterns = cPickle.load(output)
output.close()

#
# do the edge processing on the templates
#

output = open('data_iter_piy050912.pkl','rb')
train_data_iter = cPickle.load(output)
tune_data_iter = cPickle.load(output)
output.close()

train_data_iter.next()

texp = template_exp.\
    Experiment(pattern=np.array(('p','iy')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )



E,edge_feature_row_breaks,\
            edge_orientations= texp.get_edgemap_no_threshold(train_data_iter.s)

abst_threshold = np.array([0.025,0.025,0.015,0.015,
                           0.02,0.02,0.02,0.02])


del(E)
del(s)

for p in all_patterns[1:]:
    esp.threshold_edgemap(p,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
    esp.spread_edgemap(p,edge_feature_row_breaks,edge_orientations,spread_length=5)

template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns)

template_shape = np.array([template_height,template_length])
np.save('mean_template_piy050912',mean_template)
np.save('template_shape_piy050912',template_shape)
np.save('registered_templates_piy050912',registered_templates)


mean_template = np.load('mean_template_piy050912.npy')
template_shape = np.load('template_shape_piy050912.npy')

# time to get a baseline for how well the pattern applies to examples
# lets look at the max performance on the same examples with their contexts

output = open('all_patterns_context_piy050912.pkl','rb')
all_patterns_context = cPickle.load(output)
output.close()

bgd = np.load('/home/mark/projects/Template-Speech-Recognition/Experiments/042212/mean_background042212.npy')


template_length = template_shape[1]
max_detections = -np.inf * np.ones(len(all_patterns_context))

for p_id in xrange(len(all_patterns_context)):
    print p_id
    p = all_patterns_context[p_id]
    num_detections = p.shape[1] - template_length
    for d in xrange(num_detections):
        E_segment = p[:,d:d+template_length].copy()
        esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
        P,C = tt.score_template_background_section(mean_template,bgd,E_segment)
        cur_score = P+C
        if cur_score > max_detections[p_id]:
            max_detections[p_id] = cur_score


#
# want to see where the bad fits of the model are
# mainly want to see if we can improve these with the 
# parts model
#
#

max_detections_pairs = sorted([(max_detections[s],s) for s in xrange(max_detections.shape[0])])

del(all_patterns_context)

output = open('all_patterns_piy050912.pkl','rb')
all_patterns = cPickle.load(output)
output.close()

pattern_lengths = np.array([p.shape[1] for p in all_patterns])


# look at the worst performing indices
lengths_bad_examples = []
for s, p_id in max_detections_pairs[:int(len(max_detections_pairs) * .1)]:
    lengths_bad_examples.append(all_patterns[p_id].shape[1])

#
# we see that the median length is 34 which is exactly the same as the medium length for the whole dataset
# repeat bad lengths has median length 34 hence that means
# that the match works pretty well irrespective of the length, its instead variation in what the template looks like
#
# question is whether a mixture model can handle the variability
#
# some subset of these examples are going to be outliers,
# so we want to find the non outlier examples
#
#

#
# this means that the lengths of the examples is not what is not the problem, a mixture model should take care of this
#


#
# 
#
#
 


#
# create a template with the first 2/3s and the last 2/3s
#

mean_template = np.load('mean_template_piy050912.npy')
template_shape = np.load('template_shape_piy050912.npy')
bgd = np.load('/home/mark/projects/Template-Speech-Recognition/Experiments/042212/mean_background042212.npy')
registered_templates = np.load('registered_templates_piy050912.npy')


#
# registered templates testing likelihoods
#

log_template = np.log(mean_template)
log_inv_template = np.log(1-mean_template)

def compute_likelihood(log_template,log_inv_template, registered_example):
    return np.sum(log_template *registered_example + log_inv_template * (1-registered_example))
    

# apply the likelihood computation to every template
f = lambda registered_example: compute_likelihood(log_template,log_inv_template, registered_example)


registered_likelihoods = map(f,registered_templates)
likelihoods_idx_pairs = [(registered_likelihoods[idx],idx) for idx in xrange(len(registered_likelihoods))]

likelihoods_idx_pairs.sort()
#
# we are going to test a sequential version of the EM algorithm, where we add components
#
import template_speech_rec.bernoulli_em as bem
bm = bem.Bernoulli_Mixture(2,registered_templates)
bm.run_EM(.00001)

def compute_likelihood_mix(bm, registered_example):
    return max([np.sum(bm.log_templates[k] *registered_example + bm.log_invtemplates[k] * (1-registered_example)) for k in xrange(bm.log_templates.shape[0])])

f_mix = lambda registered_example: compute_likelihood_mix(bm, registered_example)

registered_likelihoods_mix = map(f_mix,registered_templates)
likelihoods_idx_pairs_mix = [(registered_likelihoods_mix[idx],idx) for idx in xrange(len(registered_likelihoods_mix))]

likelihoods_idx_pairs_mix.sort()

# we found that the likelihood computations are not really all that different

# we remove the first example because it seems to be an outlier
bm = bem.Bernoulli_Mixture(4,registered_templates[1:])
bm.run_EM(.00001)

f_mix = lambda registered_example: compute_likelihood_mix(bm, registered_example)

registered_likelihoods_mix = map(f_mix,registered_templates)
likelihoods_idx_pairs_mix = [(registered_likelihoods_mix[idx],idx) for idx in xrange(len(registered_likelihoods_mix))]

likelihoods_idx_pairs_mix.sort()


bm = bem.Bernoulli_Mixture(8,registered_templates[1:])
bm.run_EM(.00001)

f_mix = lambda registered_example: compute_likelihood_mix(bm, registered_example)

registered_likelihoods_mix = map(f_mix,registered_templates)
likelihoods_idx_pairs_mix = [(registered_likelihoods_mix[idx],idx) for idx in xrange(len(registered_likelihoods_mix))]

likelihoods_idx_pairs_mix.sort()

#
# more parts
#
#

bm = bem.Bernoulli_Mixture(40,registered_templates[1:])
bm.run_EM(.00001)

f_mix = lambda registered_example: compute_likelihood_mix(bm, registered_example)

registered_likelihoods_mix = map(f_mix,registered_templates)
likelihoods_idx_pairs_mix = [(registered_likelihoods_mix[idx],idx) for idx in xrange(len(registered_likelihoods_mix))]

likelihoods_idx_pairs_mix.sort()

#
# make a component for each training example
#

templates = np.minimum(np.maximum(registered_templates,.01),.99)
bm.log_templates = np.log(templates)
bm.log_invtemplates = np.log(1-templates)
del(templates)


f_mix = lambda registered_example: compute_likelihood_mix(bm, registered_example)

registered_likelihoods_mix = map(f_mix,registered_templates)
likelihoods_idx_pairs_mix = [(registered_likelihoods_mix[idx],idx) for idx in xrange(len(registered_likelihoods_mix))]

likelihoods_idx_pairs_mix.sort()


##
# it appears that the correct thing happens here
# we get a score of -125,06637... for every registered example
# the question is how does that transition happen as well scale up the number of points in the cluster


#
# going to try a different method to get the clustering 
# to work
#

#
#
# procedure is of the form:
#   estimate the mean only on the ones that are well on the top third of examples
#   then look at the worst third of examples
num_examples = len(likelihoods_idx_pairs)
top_third_idx = [p[1] for p in likelihoods_idx_pairs[-num_examples/3:]]

template1= np.mean(registered_templates[top_third_idx],axis=0)

bottom_twothird_idx = [p[1] for p in likelihoods_idx_pairs[:-num_examples/3]]

template2 = np.mean(registered_templates[bottom_twothird_idx],axis=0)

template1 = np.maximum(np.minimum(template1,.95),.05)
template2 = np.maximum(np.minimum(template2,.95),.05)

templates = np.array([template1,template2])
log_templates = np.log(templates)
log_invtemplates = np.log(1-templates)

bm.log_templates = log_templates
bm.log_invtemplates = log_invtemplates

f_mix = lambda registered_example: compute_likelihood_mix(bm, registered_example)

registered_likelihoods_mix = map(f_mix,registered_templates)
likelihoods_idx_pairs_mix = [(registered_likelihoods_mix[idx],idx) for idx in xrange(len(registered_likelihoods_mix))]

likelihoods_idx_pairs_mix.sort()

#
#
#
#################################################
# now we are going to try the aren template approach this means that
# we are going to divide the template into squares of side 10




bm.templates

#
#
# going to see if parts can improve on the likelihood at all
#
#

# begin by looking at the behavior of the parts on all_patterns

output = open('all_patterns_piy050912.pkl','rb')
all_patterns = cPickle.load(output)
output.close()


part_length = 2*template_shape[1]/3
parts = np.empty((2,template_shape[0],part_length))
parts[0] = mean_template[:,:part_length].copy()
parts[1] = mean_template[:,-part_length:].copy()

#function for creating the deformable template


class TwoPartModel:
    def __init__(self,template,part_length,bg):
        self.parts = np.array([template[:,:part_length],
                          template[:,-part_length:]])
        self.part_starts = np.array([0,
                                template.shape[1] - part_length])
        self.part_length = part_length
        self.get_length_range()
        self.bg = bg
        assert(parts[0].shape == parts[1].shape)
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
        self.length_range = np.array((min([p.shape[1] for p in parts]),sum([p.shape[1] for p in self.parts])))


tpm = TwoPartModel(mean_template,2*template_shape[1]/3,bgd)

tpm.get_def_template(0)
np.all(tpm.def_template[:,:template_shape[1]] == mean_template)

# basic test has been passed, now its time to test these out on the context examples

#tune_data_iter
#texp 
# both are assumed to exist


train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.7)

output = open('data_iter_piy051012.pkl','wb')
cPickle.dump(train_data_iter,output)
cPickle.dump(tune_data_iter,output)
output.close()




all_patterns = []
E_avg = template_exp.AverageBackground()            
train_data_iter.spread_length = 5

for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_patterns_context=True,
                            compute_patterns=True,
                            max_template_length=40):
        # the context length is 11
        all_patterns.extend(train_data_iter.patterns)
        E_avg.add_frames(train_data_iter.E,
                         train_data_iter.edge_feature_row_breaks,
                         train_data_iter.edge_orientations,
                         train_data_iter.abst_threshold)
    else:
        break


output = open('all_patterns_piy051012.pkl','wb')
cPickle.dump(all_patterns,output)
output.close()



for p_id in xrange(len(all_patterns)):
    esp.threshold_edgemap(all_patterns[p_id],.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
    esp.spread_edgemap(all_patterns[p_id],edge_feature_row_breaks,edge_orientations,spread_length=5)






template_height, template_length, \
    registered_examples,mean_template \
    = et.simple_estimate_template(all_patterns)

mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)


np.save('mean_template_piy051012',mean_template)
np.save('registered_examples_piy051012',registered_examples)
np.save('mean_background_piy051012',mean_background)
# free up some memory for the rest of the experiment
del(all_patterns)

tune_data_iter.reset_exp()
tuning_patterns_context = []



for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                            compute_patterns_context=True,
                            max_template_length=44):
        # the context length is 11
        tuning_patterns_context.extend(tune_data_iter.patterns_context)
    else:
        break


# find the optimal lengths

tuning_pattern_times = []
tune_data_iter.reset_exp()

for datum_id in xrange(tune_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if tune_data_iter.next(wait_for_positive_example=True,
                            compute_pattern_times=True,
                            max_template_length=44):
        # the context length is 11
        tuning_pattern_times.extend(tune_data_iter.pattern_times)
    else:
        break




#
# get a baseline score
#
#
# somethign to look at later is to see the  counts for all
# disyllables
#

tpm = TwoPartModel(mean_template,2*template_shape[1]/3,mean_background)

def_range = np.arange(-6,17)

tpm.get_def_template(0)

all_def_templates = np.empty((def_range.shape[0],
                            tpm.def_template.shape[0],
                            tpm.def_template.shape[1]))
for d in xrange(def_range.shape[0]):
    tpm.get_def_template(def_range[d])
    all_def_templates[d] = tpm.def_template.copy()
    

base_detection_scores = np.empty(len(tuning_patterns_context))
optimal_detection_scores = np.empty(len(tuning_patterns_context))
true_length_detection_scores = np.empty(len(tuning_patterns_context))
optimal_length = np.empty(len(tuning_patterns_context))
for c_id in xrange(len(tuning_patterns_context)):
    cur_context = tuning_patterns_context[c_id]
    num_detections = cur_context.shape[1] - tpm.length_range[1]
    win_length = tpm.length_range[1]
    for d in xrange(num_detections):
        E_window = cur_context[:,d:d+win_length].copy()
        esp.threshold_edgemap(E_window,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_window,edge_feature_row_breaks,edge_orientations,spread_length=3)
        # base detection
        def_template = all_def_templates[6]
        P,C = tt.score_template_background_section(def_template,tpm.bg,E_window)
        for d in range
        esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                                                   
                                                   

max_defs = get_max_defs(parts)


def_template = get_def_template(parts,bgd,def_size)



def get_def_template(parts,bgd,def_size):
    max_defs = get_max_defs(parts)
    assert ( max_defs[0] <= def_size

#
# First test that the deformable template does better
#

ma
