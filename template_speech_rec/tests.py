##########################
#
#  parts_model.py
#
#
##########################

import sys, os
sys.path.append('/home/mark/projects/Template-Speech-Recognition')
import template_speech_rec.template_experiments as t_exp
import template_speech_rec.parts_model as pm
import template_speech_rec.template_experiments as t_exp
reload(t_exp)
import numpy as np
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.process_scores as ps
from pylab import imshow, plot, figure, show
import template_speech_rec.bernoulli_em as bem
import template_speech_rec.parts_model as pm
import cPickle



data_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/042212/'

mean_template = np.load(data_dir + 'mean_template042212.npy')
#bgd = np.load(data_dir + 'mean_background042212')
bgd = .4 * np.ones(mean_template.shape[0])

parts_model = pm.PartsTriple(mean_template,.4,.4,.4)



t1 = parts_model.front_length == parts_model.front.shape[1]
t2 = parts_model.front_length-parts_model.front_def_radius == parts_model.middle_start
t3 = np.all(parts_model.front[:,:parts_model.front_length] == mean_template[:,:int(.4*mean_template.shape[1])])
t4 = np.all(parts_model.back[:,-parts_model.back_length:] == mean_template[:,-int(.4*mean_template.shape[1]):])
#
# check that the deformed_max_length is correct
#
#

t5 = parts_model.deformed_max_length -parts_model.front_def_radius - parts_model.back_def_radius == mean_template.shape[1]


# broke this agggghhhhh!
input_fl = open(data_dir + 'part_testing042212.pkl','rb')
save_dict = cPickle.load(input_fl)
input_fl.close()

bg_len=save_dict['bg_len']
E=save_dict('E')
phns=save_dict('phns')
edge_feature_row_breaks=save_dict('edge_feature_row_breaks')
edge_orientations=save_dict('edge_orientations')
phn_times=save_dict('phn_times')
exp=save_dict('exp')
phn_times=save_dict('phn_times')
pattern_times=save_dict('pattern_times')
patterns=save_dict('patterns')
bgds=save_dict('bgds')
s=save_dict('s')


reload(pm)
parts_model = pm.PartsTriple(mean_template,.4,.4,.4)

front_displace = 3
back_displace = -1
deformed_template = parts_model.get_deformed_template(front_displace,back_displace,bgd)

#
# first test is just that the first frames and last frames
#  as determined by the deformation radius should be background

def_template = parts_model.get_deformed_template(0,0,bgd)
t1 = def_template

parts_model.fit_template(E,E_loc,front_displace,back_displace)



################################
#
#  template_experiments.py
#
#################################

E_copy = E.copy()
#E
template=mean_template
bgd_length=26
mean_background=mean_bgd
#edge_feature_row_breaks
#edge_orientations
abst_threshold=abst_threshold
spread_length=3



pbgd = E[:,pattern_times[0][0]-bg_len:pattern_times[0][0]].copy()
pbgd2 = pbgd.copy()
esp.threshold_edgemap(pbgd2,.30,edge_feature_row_breaks,
                              abst_threshold=abst_threshold)
esp.spread_edgemap(pbgd2,edge_feature_row_breaks,edge_orientations,
                           spread_length=spread_length)
pbgd2 = np.mean(pbgd2,axis=1)
pbgd2 = np.maximum(np.minimum(pbgd2,.4),.1)

template_height,template_length = template.shape
num_detections = E.shape[1]-template_length+1
E_background, estimated_background_idx = self._get_E_background(E,num_detections,bgd_length, mean_background,
                                                                edge_feature_row_breaks,
                                                                edge_orientations,
                                                                abst_threshold=abst_threshold,
                                                                spread_length=spread_length)


# found that 
for frame_id in xrange(bg_len,E_background.shape[1]):
    bg_seg = E[:,frame_id-bg_len:frame_id].copy()
    esp.threshold_edgemap(bg_seg,.30,edge_feature_row_breaks,
                              abst_threshold=abst_threshold)
    esp.spread_edgemap(bg_seg,edge_feature_row_breaks,edge_orientations,
                           spread_length=spread_length)
    bg_seg = np.mean(bg_seg,axis=1)
    bg_seg = np.maximum(np.minimum(bg_seg,.4),.1)
    print np.all(bg_seg == E_background[:,frame_id])



# verified that the background computation is reasonable



###################
# bernoulli_em.py
###################

import sys, os
#sys.path.append('/var/tmp/stoehr/Projects/Template-Speech-Recognition')
sys.path.append('/home/mark/projects/Template-Speech-Recognition')

import template_speech_rec.bernoulli_em as bem
import numpy as np
import random
reload(bem)

a = np.arange(24).reshape(2,3,4)
bm = bem.bernoulli_EM(a,1,1)


bm = bem.bernoulli_EM(a,2,.1)
assert(bm.get_templates().shape[0] == 2)
assert(bm.get_templates().shape[1:]== a.shape[1:])


T = np.random.rand(9*2).reshape(2,3,3)

tlike_func = np.vectorize(lambda datum: \
                              np.prod(T[:,datum > .5],
                                      axis=1) *\
                              np.prod((1-T)[:,datum < .5],
                                      axis=1))

datum = np.random.rand(9).reshape(3,3)
data_mat = np.random.rand(10*9).reshape(10,3,3)


random.seed()
idx = range(data_mat.shape[0])
random.shuffle(idx)
num_mix = 2
bm = bem.Bernoulli_Mixture(data_mat.shape[1:],num_mix)
bm.set_templates(data_mat[idx])
likelihood = -np.inf
new_likelihood,E_mat = bm.compute_likelihoods(data_mat)
while new_likelihood - likelihood > tol:
    likelihood = new_likelihood
    # E-step
    bm.update_weights(get_E_weights(E_mat,data_mat))
    # M_step
    bm.update_templates(template_weights,data_mat)
    new_likelihood,E_mat = bm.compute_likelihood(data_mat)
        

##
#
# Generate Synthetic data see if the mixtures can pick
# up on it
#

import template_speech_rec.bernoulli_em as bem
import numpy as np
import random

### Just do 0s and 1s at first

true_templates = np.random.rand(3*3*2).reshape(2,3,3)
true_weights = np.random.rand(2)
data_mat = np.zeros((500,true_templates.shape[1],true_templates.shape[2]))



vals = np.random.rand(data_mat.size).reshape(data_mat.shape)
for k in xrange(vals.shape[0]):
    use_component = np.random.multinomial(1,true_weights)
    data_mat[k][vals[k]<true_templates[use_component==1][0]] = 1

data_ma
a = np.ones(100)
a[vals <.7] = 0.

#
# mixture centers should be 1 and 0, weights .3 and .7 respectively
#

a =a.reshape(100,1,1)

bm = bem.bernoulli_EM(a,2,.000001)

data_mat = a
num_mix = 2
bm = bem.Bernoulli_Mixture(num_mix,data_mat)

# testing compute_loglikelihoods
template_scores = bm.get_template_likelihoods(bm.data_mat)
likelihoods = template_scores * np.tile(bm.weights,(bm.num_data,1))
marginals = np.sum(likelihoods,axis=1)
loglikelihood = np.sum(np.log(marginals))
affinities =likelihoods/np.tile(marginals,
                   (bm.num_mix,1)).transpose()

##
# Testing the M step
#

weights = np.mean(affinities,axis=0)
for mix_id in xrange(bm.num_mix):
    templates[mix_id] = np.sum(bm.data_mat * np.tile(affinities[:,mix_id],(bm.width,bm.height,1)).transpose(),axis=0)
    templates[mix_id] = templates[mix_id]/(weights[mix_id] * bm.num_data)



np.dot(template_scores,self.weights), template_scores






if num_mix == 1:
    bm.set_templates(clean_templates(np.mean(data_mat,axis=0).reshape(1,data_mat.shape[1],data_mat.shape[2])))
    bm.set_weights(np.array([1.]))
elif num_mix > 1:
    run_EM(bm,data_mat,num_mix,tol)
return bm


random.seed()
idx = range(data_mat.shape[0])
random.shuffle(idx)
bm.set_templates(data_mat[idx])
loglikelihood = -np.inf
new_likelihood,E_mat = bm.compute_likelihoods(data_mat)
new_loglikelihood = np.sum(np.log(new_likelihood))
while new_loglikelihood - loglikelihood > tol:
    loglikelihood = new_loglikelihood
    # E-step
    bm.update_weights(get_E_weights(E_mat,data_mat))
    # M_step
    bm.update_templates(template_weights,data_mat)
    new_likelihood,E_mat = bm.compute_likelihood(data_mat)
    new_loglikelihood = np.sum(np.log(new_likelihood))


#
# Testing on registered templates
#
#

registered_templates = np.load('/home/mark/projects/Template-Speech-Recognition/Experiments/RedoAlexey/registered_templates041312.npy')


bm = bem.Bernoulli_Mixture(num_mix,registered_templates)
bm.run_EM(.001)


template_logscores = bm.get_template_loglikelihoods(bm.data_mat)
loglikelihoods = template_logscores + np.tile(np.log(bm.weights),(bm.num_data,1))
max_vals = np.amax(loglikelihoods,axis=1)
# adjust the marginals by a value to avoid numerical
# problems
logmarginals_adj = np.sum(np.exp(loglikelihoods - np.tile(max_vals,(2,1)).transpose()),axis=1)
loglikelihood = np.sum(np.log(logmarginals_adj)) + np.sum(max_vals)
affinities =loglikelihoods-np.tile(logmarginals_adj+max_vals,
                   (bm.num_mix,1)).transpose()
affinities/=np.tile(np.sum(affinities,axis=1),(bm.num_mix,1)).transpose()

bm.run_EM(.00001)
loglikelihood = -np.inf
new_loglikelihood = bm.compute_loglikelihoods()


datum = bm.data_mat[0].copy()
log_template_score = np.sum(np.log(bm.templates[:,datum > .5]),\
                            axis=1) +\
                            np.sum(np.log((1-bm.templates)[:,datum < .5]),\
                                       axis=1)

template_logscores = np.array(map(lambda x:\
                 bm.get_template_loglikelihood(x),\
                 bm.data_mat))

loglikelihoods = template_logscores + np.tile(np.log(bm.weights),(bm.num_data,1))

max_vals = np.amax(loglikelihoods,axis=1)
marginals = np.sum(np.exp(loglikelihoods - np.tile(max_vals,(2,1)).transpose()),axis=1)

loglikelihood = np.sum(np.log(marginals)) + np.sum(max_vals)

template_scores = bm.get_template_likelihoods(bm.data_mat)
likelihoods = template_scores * np.tile(bm.weights,(bm.num_data,1))
marginals = np.sum(likelihoods,axis=1)
loglikelihood = np.sum(np.log(marginals))
affinities =likelihoods/np.tile(marginals,
                                     (bm.num_mix,1)).transpose()



while new_loglikelihood - loglikelihood > tol:
    loglikelihood = new_loglikelihood
    # M-step
    bm.M_step()
    # E-step
    new_loglikelihood = bm.compute_loglikelihoods()
 


#
# Full EM experiment now that
#
#
true_templates = .05 * np.ones((3,6,6))
true_templates[0,2,:] = .85
true_templates[1,:,3] = .85
true_templates[2,4,:] = .85

true_weights = np.random.rand(3)
true_weights /= np.sum(true_weights)
data_mat = np.zeros((100,true_templates.shape[1],true_templates.shape[2]))

vals = np.random.rand(data_mat.size).reshape(data_mat.shape)
for k in xrange(vals.shape[0]):
    use_component = np.random.multinomial(1,true_weights)
    data_mat[k][vals[k]<true_templates[use_component==1][0]] = 1

bm = bem.Bernoulli_Mixture(3,data_mat)
bm.run_EM(.00001)


