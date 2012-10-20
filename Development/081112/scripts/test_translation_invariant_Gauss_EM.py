import numpy as np
from scipy import correlate
from scipy.linalg import circulant

def generate_all_shifts(atom, signal_atom_diff):
    return circulant(np.hstack((atom, np.zeros(signal_atom_diff)))).T[:signal_atom_diff+1]

#s=np.sin(np.arange(100)/99. * 10.*np.pi)
s = np.arange(12)

all_shifts=generate_all_shifts(s,10)
use_shifts = all_shifts[::4]
data_mat = np.tile(use_shifts,(20,1)) + np.random.randn(60,use_shifts.shape[1]) * .05


component_length = 12
num_data,data_length = data_mat.shape
trans_amount = data_length-component_length +1 
num_mix = 1
affinities = np.zeros((num_mix,num_data,trans_amount))
rand_mix_idx = np.random.randint(num_mix,
                                         size=(num_data))
affinities[rand_mix_idx,
           np.arange(num_data),
                        np.zeros(num_data,
                                 dtype=int)] = 1.

np.array([
                np.array([
                        correlate(datum,affinity_row)
                        for affinity_row in affinity])
                for datum, affinity in zip(data_mat,
                                           affinities)])


marginalized_translations = np.array([
        np.array([ correlate(datum,affinity_row)
                  for datum, affinity_row in zip(data_mat,
                                                 component_affinities)])
        for component_affinities in affinities])

means = marginalized_translations.mean(1)
covs = ((marginalized_translations - means)**2).mean(1)
# 
norm_constants = -.5 * np.log((2.*np.pi)**num_mix * np.prod(covs,1))
from translation_invariant_Gauss_EM import TranslationInvariantGaussEM
tig = TranslationInvariantGaussEM(1,data_mat,12)
tig.run_EM(.0001)

trans_data_mat = np.array([
                generate_shifts(datum,trans_amount)
                for datum in data_mat])

for cur_component, vals in enumerate(zip(means,
                                         covs,
                                         norm_constants)):
    mean, cov, norm_constant = vals
    affinities[cur_component] = np.sum((trans_data_mat - mean)**2 * cov,2) + norm_constant


#
#

#

from _translation_invariant_Gauss_EM import subtract_max_affinities, normalize_affinities

affinities *=2

normalize_affinities(affinities,np.sum(affinities.sum(0),1),
                     num_mix, num_data, trans_amount)

subtract_max_affinities(affinities,np.sum(affinities.sum(0),1),
                     num_mix, num_data, trans_amount)




import translation_invariant_Gauss_EM 
reload(translation_invariant_Gauss_EM)
TranslationInvariantGaussEM = translation_invariant_Gauss_EM.TranslationInvariantGaussEM
tig = TranslationInvariantGaussEM(1,data_mat,12)
tig.run_EM(.0001)

datamat = np.zeros((100,10))
component_length = 9
for i in xrange(100):
    if i % 2 == 0:
        datamat[i,:component_length] = 100 * np.ones(component_length) 
    else:
        datamat[i,-component_length:] = 100*np.ones(component_length)

datamat += np.random.randn(100,10) * .05

tig = TranslationInvariantGaussEM(1,datamat,9)
tig.run_EM(.0001)


def generate_shifts(atom, num_shifts):
    if num_shifts == 1:
        return np.array([atom])
    else:
        return circulant(np.hstack((atom, np.zeros(num_shifts-1)))).T[:num_shifts][:,num_shifts-1:1-num_shifts]



num_mix = 1
data_mat = data_mat
num_data, data_length = data_mat.shape
# to make likelihood computation go faster
rep_data_mat = np.tile(data_mat.reshape(num_data,
                                                  1,
                                                  data_length),
                            (1,
                             num_mix,
                             1))
assert data_mat.ndim == 2
component_length = component_length
trans_amount = data_length - component_length + 1
# shifted versions of the data
trans_data_mat = np.array([
        generate_shifts(datum,trans_amount)
        for datum in data_mat])
affinities = np.zeros((num_mix,
                            num_data,
                            trans_amount))
max_affinities = np.zeros(affinities.shape[:-1])
# initialize variables as None so that we know they are
# defined later
means = None
covs = None
norm_constants = None
mix_weights = None
log_likelihood = - np.inf
# uniform weighting over the transitions
trans_weights = np.ones(trans_amount,dtype=np.float64)
trans_weights /= np.sum(trans_weights)

affinities[:] = 0
rand_mix_idx = np.random.randint(num_mix,
                                 size=(num_data))
affinities[rand_mix_idx,
                np.arange(num_data),
                np.zeros(num_data,
                         dtype=int)] = 1.
mix_weights = np.array([np.sum(rand_mix_idx==i) for i in xrange(num_mix)]).astype(np.float32)
mix_weights /= np.sum(mix_weights)


marginalized_translations = np.array([
        np.array([ correlate(datum,affinity_row)
                   for datum, affinity_row in zip(data_mat,
                                                  component_affinities)])
        for component_affinities in affinities])
means = marginalized_translations.mean(1)
covs = ((marginalized_translations - means)**2).mean(1)
# 
norm_constants = -.5 * np.log((2.*np.pi)**num_mix * np.prod(covs,1))



for cur_component, vals in enumerate(zip(means,
                                         covs,
                                         norm_constants)):
    mean, cov, norm_constant = vals
    affinities[cur_component] = np.sum(-.5*(trans_data_mat - mean)**2 * cov,2) + norm_constant    
    max_affinities[cur_component] =np.max(affinities[cur_component],1)

affinities = np.exp(affinities)
log_likelihood = np.log(np.sum(
        np.sum(
            affinities * trans_weights,
            2).T * mix_weights,1))

affinities /= np.tile(
    np.sum(np.sum(affinities,2),0).reshape(1,
                                                num_data,
                                                1),
    (num_mix,1,trans_amount))

mix_weights = np.sum(np.sum(affinities,1),1)
mix_weights /= np.sum(mix_weights)



import numpy as np
import pylab as pl
from sklearn.svm import SVC
from sklearn.preprocessing import Scaler
from sklearn.datasets import load_iris
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold


import cPickle, os
root_dir = '/home/mark/Template-Speech-Recognition/'
data_dir = root_dir + 'Experiments/081112/data/'

feature_file_suffix = 'feature_list.npy'

feature_file_names = [
    feature_file_name
    for feature_file_name in os.listdir(data_dir)
    if feature_file_name[-len(feature_file_suffix):] == feature_file_suffix]

feature_file_idx = 0
# Generate train data
X = np.array([np.array(l) for l in np.load(data_dir+feature_file_names[feature_file_idx]) if len(l) ==640])

num_mix =1

num_mix = num_mix
data_mat = X[:300]
num_data, data_length = data_mat.shape
# to make likelihood computation go faster
rep_data_mat = np.tile(data_mat.reshape(
        num_data,
        1,
                data_length),
                            (1,
                             num_mix,
                             1))
assert data_mat.ndim == 2
component_length = component_length
trans_amount = data_length - component_length + 1
# shifted versions of the data
trans_data_mat = np.array([
        np.array([
                correlate(datum,unit_vec)
                for unit_vec in np.eye(trans_amount)])
        for datum in data_mat])
affinities = np.zeros((num_mix,
                            num_data,
                            trans_amount))
# initialize variables as None so that we know they are
# defined later
means = None
covs = None
norm_constants = None
mix_weights = None
log_likelihood = - np.inf
# uniform weighting over the transitions
trans_weights = np.ones(trans_amount,dtype=np.float64)
trans_weights /= np.sum(trans_weights)

affinities[:] = 0
rand_mix_idx = np.random.randint(num_mix,
                                 size=(num_data))
affinities[rand_mix_idx,
                np.arange(num_data),
                np.zeros(num_data,
                         dtype=int)] = 1.
mix_weights = np.array([np.sum(rand_mix_idx==i) for i in xrange(num_mix)]).astype(np.float32)
mix_weights /= np.sum(mix_weights)
norm_constant_constant =  np.log((2.*np.pi)**num_mix)

max_affinities = np.zeros(
    num_data)


marginalized_translations = np.array([
        np.array([ correlate(datum,affinity_row)
                   for datum, affinity_row in zip(data_mat,
                                                  component_affinities)])
        for component_affinities in affinities])
means = marginalized_translations.mean(1)
covs = ((marginalized_translations - means)**2).mean(1)
        # 
norm_constants = norm_constant_constant -.5 * np.sum(norm_constant_constant+np.log(covs),1)




for cur_component, vals in enumerate(zip(means,
                                                 covs,
                                                 norm_constants)):
    mean, cov, norm_constant = vals
    affinities[cur_component] = np.sum(- .5 *(trans_data_mat - mean)**2 * cov,2) + norm_constant
max_affinities[:] = np.max(
    np.max(affinities,
           0),
    1)
subtract_max_affinities(affinities,max_affinities,
                        num_mix,
                        num_data,
                        trans_amount)
affinities = np.exp(affinities)
log_likelihood = np.sum(np.log(np.sum(
            np.sum(
                affinities * trans_weights,
                2).T * mix_weights,1)) + max_affinities)
affinity_sums = np.sum(affinities.sum(0),1)
normalize_affinities(affinities,affinity_sums,
                             num_mix,
                             num_data,
                             trans_amount)
mix_weights = np.sum(np.sum(affinities,1),1)
mix_weights /= np.sum(mix_weights)

        
