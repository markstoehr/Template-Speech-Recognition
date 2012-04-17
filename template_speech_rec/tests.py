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


