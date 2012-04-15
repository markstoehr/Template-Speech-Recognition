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

true_templates = np.array([[[.2,.8]],[[.5,.1]]])
true_weights = np.array([.5,.5])
data_mat = np.zeros((100,true_templates.shape[1],true_templates.shape[2]))



vals = np.random.rand(data_mat.size).reshape(data_mat.shape)
for k in xrange(vals.shape[0]):
    idx = k % 2
    data_mat[k][vals[k]>true_templates[idx]] = 1

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
