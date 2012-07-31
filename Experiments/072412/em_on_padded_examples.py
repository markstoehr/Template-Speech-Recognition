import numpy as np
from collections import defaultdict
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
exp_path = root_path+'Experiments/072412/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bernoulli_mixture

p_examples5 = np.load(data_path+'pclass_examples5.npy')
p_lengths = np.load(data_path+'pclass_examples_lengths.npy')
p_bgs = np.load(data_path+'pclass_examples_bgs.npy')

# TODO: sample from edges


def make_padded_example(example,length,bg):
    residual_length = example.shape[1] - length
    height = len(bg)
    if residual_length > 0:
        return np.hstack(
            (example[:,:length],
             (np.random.rand(height,residual_length) <
             np.tile(bg,(residual_length,
                         1)).T).astype(np.uint8)))
    return example
         
def make_padded(examples,lengths,bgs):
    return np.array([
            make_padded_example(example,length,bg)\
            for example,length, bg in \
            zip(examples,lengths,bgs)])


p_padded = make_padded(p_examples5,
                       p_lengths,
                       p_bgs)


bm2 = bernoulli_mixture.BernoulliMixture(2,p_padded)
bm2.run_EM(.00001)
#
# want to save the templates
# also want to see if the affinities
# in any way relate to the lengths of the templates
bm2_t_0_lengths = p_lengths[bm2.affinities[:,0] ==1]
bm2_t_1_lengths = p_lengths[bm2.affinities[:,1] ==1]

"""
Mixtures not picking up on the example lengths

>>> bm2_t_0_lengths.mean()
8.8964451313755788
>>> bm2_t_1_lengths.mean()
8.7279752704791349
"""


bm3 = bernoulli_mixture.BernoulliMixture(3,p_padded)
bm3.run_EM(.00001)

bm3_t_0_lengths = p_lengths[bm3.affinities[:,0] ==1]
bm3_t_1_lengths = p_lengths[bm3.affinities[:,1] ==1]
bm3_t_2_lengths = p_lengths[bm3.affinities[:,2] ==1]

"""
Mixtures seem to picking up on the lengths of the examples:

>>> bm3_t_2_lengths.mean()
5.5180722891566267
>>> bm3_t_1_lengths.mean()
13.87034035656402
>>> bm3_t_0_lengths.mean()
9.6352112676056336


>>> np.median(bm3_t_0_lengths)
9.0
>>> np.median(bm3_t_1_lengths)
14.0
>>> np.median(bm3_t_2_lengths)
5.0

"""

bm4 = bernoulli_mixture.BernoulliMixture(4,p_padded)
bm4.run_EM(.00001)

for i in xrange(4):
    print p_lengths[bm4.affinities[:,i] ==1].mean()
    
"""
from a length perspective we may not be getting much out
of the multiple templates

>>> for i in xrange(4):
    print p_lengths[bm4.affinities[:,i] ==1].mean()
    

... ... ... 13.9005145798
9.38771186441
4.69667738478
9.83300589391

>>> for i in xrange(4):
    print i, int(np.median(p_lengths[bm4.affinities[:,i] ==1]))
... ... 
0 14
1 9
2 5
3 10


"""
