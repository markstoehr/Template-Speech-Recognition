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
p_train_masks = np.load(exp_path+'p_train_masks.npy')
p_utt_id_E_loc = np.load(data_path+'pexample_utt_id_E_loc.npy')
p_specs = np.zeros((p_bgs.shape[0],
                    p_bgs.shape[1]/8,
                    p_examples5.shape[2]),dtype=np.float32)

# store the spectrograms of the examples
for p_id, p_utt in enumerate(p_utt_id_E_loc):
    p_example = esp.get_spectrogram_features(
        np.load(data_path+str(p_utt[0]+1)+'s.npy'),16000,320,80,512,3000,7)[
        :,
        p_utt[1]:p_utt[1]+p_specs.shape[2]]
    p_specs[p_id][:,:p_example.shape[1]] = p_example
 
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


# need to construct a function now that performs averages over the 10 folds
# to make sure that I have across fold consistency for these estimates
# plan for comparison is to output the means for each 
#
#

class SpecMean:
    def __init__(self,use_length,height):
        self.S = np.zeros((height,use_length),dtype=np.float32)
        self.height = height
        self.use_length = use_length
        self.num_E = 0.
    def add_new_example(self,example):
        if example.shape[1] < self.use_length:
            example = np.hstack((example,np.zeros((self.height,self.use_length-example.shape[1]))))
        self.num_E += 1.
        delta = example - self.S
        self.S = self.S + delta/self.num_E

def construct_spectrogram_mean(utt_id_E_loc,data_path,use_length,height):
    smean = SpecMean(use_length,height)
    [smean.add_new_example(
            esp.get_spectrogram_features(
                np.load(data_path+str(x[0])+'s.npy'),16000,320,80,512,3000,7)[
                :,
                x[1]:x[1]+use_length])
            for x in utt_id_E_loc]
    return smean
    
    
import collections
BernoulliMixtureSimple = collections.namedtuple('BernoulliMixtureSimple',
                                                'log_templates log_invtemplates weights num_mix')


def bernoulli_likelihood(log_template,
                         log_invtemplate,
                         data_mat):
    return np.sum(np.tile(log_template,
                     (data_mat.shape[0],1)) * data_mat +
             np.tile(log_invtemplate,
                     (data_mat.shape[0],1)) * (1-data_mat),1)


    
def bernoulli_model_loglike(bernoulli_model,data_mat,
                            use_weights=False):
    data_dim = data_mat.shape[1:]
    num_data = data_mat.shape[0]
    if len(data_dim) > 1:
        data_mat = data_mat.reshape(num_data,np.prod(data_dim))
    bernoulli_out = np.zeros((bernoulli_model.num_mix,
                              data_mat.shape[0]
                              ))
    for mix_id in xrange(bernoulli_model.num_mix):
        bernoulli_out[mix_id] = bernoulli_likelihood(bernoulli_model.log_templates[mix_id],
                                                     bernoulli_model.log_invtemplates[mix_id],
                                                     data_mat)
    if use_weights:
        bernoulli_out = bernoulli_out *np.tile(bernoulli_model.weights,
                                               (num_data,1)).T
        return np.sum(bernoulli_out)
    if len(data_dim) > 1:
        data_mat = data_mat.reshape((num_data,) + data_dim)
    return np.sum(np.max(bernoulli_out,0))
                                                     
                                  
                                  

# these are the mean and median lengths of the examples that are most
# strongly associated with a particular mixture component so the affinity
# is greater that .99
bernoulli_mixture_induced_lengths = []
bernoulli_mixture_spec_avgs = []
dev_likelihoods = np.zeros((5,len(p_train_masks)))
example_cluster_ids = []
for mix_id, k in enumerate([2,3,4,5]):
    example_cluster_ids_fold = []
    print k
    for train_mask_id , train_mask in enumerate(p_train_masks):
        medians_vec = np.zeros(k)
        mean_vec = np.zeros(k)
        spec_avgs = np.zeros((k,p_specs.shape[1], p_specs.shape[2]),dtype=np.float32)
        spec_meds = np.zeros((k,p_specs.shape[1], p_specs.shape[2]),dtype=np.float32)
        E_avgs = np.zeros((k,p_examples5.shape[1], p_examples5.shape[2]),dtype=np.float32)
        E_meds = np.zeros((k,p_examples5.shape[1], p_examples5.shape[2]),dtype=np.float32)
        print train_mask_id
        bm = bernoulli_mixture.BernoulliMixture(k,p_padded[train_mask])
        bm.run_EM(.00001)
        bernoulli_model = BernoulliMixtureSimple(
            log_templates=bm.log_templates,
            log_invtemplates=bm.log_invtemplates,
            weights=bm.weights,
            num_mix=k)
        dev_likelihoods[mix_id+1,train_mask_id] = bernoulli_model_loglike(bernoulli_model,p_padded[True-train_mask],
                                                                          use_weights=False)
        cluster_ids = []
        for i in xrange(k):
            cluster_ids.append(bm.affinities[:,i] > .99)
            mix_lengths = p_lengths[bm.affinities[:,i] >.99]
            medians_vec[i] = np.median(mix_lengths)
            mean_vec[i] = np.mean(mix_lengths)
            spec_avgs[i] = p_specs[bm.affinities[:,i] > .99].mean(0)
            spec_meds[i] = np.median(p_specs[bm.affinities[:,i] > .99],axis=0)
            E_avgs[i] = np.mean(p_examples5[bm.affinities[:,i] > .99],axis=0)
            E_meds[i] = np.median(p_examples5[bm.affinities[:,i] > .99],axis=0)
        example_cluster_ids_fold.append(tuple(cluster_ids))
        save_str = ('/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/080212/'
                + str(k)+'_'+str(train_mask_id)+'_')
        np.save(save_str+'p_medians_vec',medians_vec)
        np.save(save_str+'p_means_vec',mean_vec)
        np.save(save_str+'p_spec_avgs',spec_avgs)
        np.save(save_str+'p_spec_meds',spec_meds)
        np.save(save_str+'p_E_avgs',E_avgs)
        np.save(save_str+'p_E_meds',E_meds)
        np.save(save_str+'p_templates',bm.templates           )
    example_cluster_ids.append(tuple(example_cluster_ids_fold))



dev_likelihoods2 = np.zeros((3,len(p_train_masks)))
example_cluster_ids2 = []
for mix_id, k in enumerate([6,8,10]):
    example_cluster_ids_fold = []
    print k
    for train_mask_id , train_mask in enumerate(p_train_masks):
        medians_vec = np.zeros(k)
        mean_vec = np.zeros(k)
        spec_avgs = np.zeros((k,p_specs.shape[1], p_specs.shape[2]),dtype=np.float32)
        spec_meds = np.zeros((k,p_specs.shape[1], p_specs.shape[2]),dtype=np.float32)
        E_avgs = np.zeros((k,p_examples5.shape[1], p_examples5.shape[2]),dtype=np.float32)
        E_meds = np.zeros((k,p_examples5.shape[1], p_examples5.shape[2]),dtype=np.float32)
        print train_mask_id
        bm = bernoulli_mixture.BernoulliMixture(k,p_padded[train_mask])
        bm.run_EM(.00001)
        bernoulli_model = BernoulliMixtureSimple(
            log_templates=bm.log_templates,
            log_invtemplates=bm.log_invtemplates,
            weights=bm.weights,
            num_mix=k)
        dev_likelihoods2[mix_id,train_mask_id] = bernoulli_model_loglike(bernoulli_model,p_padded[True-train_mask],
                                                                          use_weights=False)
        cluster_ids = []
        for i in xrange(k):
            cluster_ids.append(bm.affinities[:,i] > .99)
            mix_lengths = p_lengths[bm.affinities[:,i] >.99]
            medians_vec[i] = np.median(mix_lengths)
            mean_vec[i] = np.mean(mix_lengths)
            spec_avgs[i] = p_specs[bm.affinities[:,i] > .99].mean(0)
            spec_meds[i] = np.median(p_specs[bm.affinities[:,i] > .99],axis=0)
            E_avgs[i] = np.mean(p_examples5[bm.affinities[:,i] > .99],axis=0)
            E_meds[i] = np.median(p_examples5[bm.affinities[:,i] > .99],axis=0)
        example_cluster_ids_fold.append(tuple(cluster_ids))
        save_str = ('/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/080212/'
                + str(k)+'_'+str(train_mask_id)+'_')
        np.save(save_str+'p_medians_vec',medians_vec)
        np.save(save_str+'p_means_vec',mean_vec)
        np.save(save_str+'p_spec_avgs',spec_avgs)
        np.save(save_str+'p_spec_meds',spec_meds)
        np.save(save_str+'p_E_avgs',E_avgs)
        np.save(save_str+'p_E_meds',E_meds)
        np.save(save_str+'p_templates',bm.templates           )
    example_cluster_ids2.append(tuple(example_cluster_ids_fold))


dev_likelihoods3 = np.zeros((7,len(p_train_masks)))
example_cluster_ids3 = []
for mix_id, k in enumerate([2,3,4,5,6,8,10]):
    example_cluster_ids_fold = []
    print k
    for train_mask_id , train_mask in enumerate(p_train_masks):
        medians_vec = np.zeros(k)
        mean_vec = np.zeros(k)
        spec_avgs = np.zeros((k,p_specs.shape[1], p_specs.shape[2]),dtype=np.float32)
        spec_meds = np.zeros((k,p_specs.shape[1], p_specs.shape[2]),dtype=np.float32)
        E_avgs = np.zeros((k,p_examples5.shape[1], p_examples5.shape[2]),dtype=np.float32)
        E_meds = np.zeros((k,p_examples5.shape[1], p_examples5.shape[2]),dtype=np.float32)
        print train_mask_id
        bm = bernoulli_mixture.BernoulliMixture(k,p_padded[train_mask])
        bm.run_EM(.00001)
        bernoulli_model = BernoulliMixtureSimple(
            log_templates=bm.log_templates,
            log_invtemplates=bm.log_invtemplates,
            weights=bm.weights,
            num_mix=k)
        dev_likelihoods3[mix_id,train_mask_id] = bernoulli_model_loglike(bernoulli_model,p_padded[True-train_mask],
                                                                          use_weights=True)
        cluster_ids = []
        for i in xrange(k):
            cluster_ids.append(bm.affinities[:,i] > .99)
            mix_lengths = p_lengths[bm.affinities[:,i] >.99]
            medians_vec[i] = np.median(mix_lengths)
            mean_vec[i] = np.mean(mix_lengths)
            spec_avgs[i] = p_specs[bm.affinities[:,i] > .99].mean(0)
            spec_meds[i] = np.median(p_specs[bm.affinities[:,i] > .99],axis=0)
            E_avgs[i] = np.mean(p_examples5[bm.affinities[:,i] > .99],axis=0)
            E_meds[i] = np.median(p_examples5[bm.affinities[:,i] > .99],axis=0)
        example_cluster_ids_fold.append(tuple(cluster_ids))
        save_str = ('/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/080212/'
                + str(k)+'_'+str(train_mask_id)+'_')
        np.save(save_str+'p_medians_vec',medians_vec)
        np.save(save_str+'p_means_vec',mean_vec)
        np.save(save_str+'p_spec_avgs',spec_avgs)
        np.save(save_str+'p_spec_meds',spec_meds)
        np.save(save_str+'p_E_avgs',E_avgs)
        np.save(save_str+'p_E_meds',E_meds)
        np.save(save_str+'p_templates',bm.templates           )
    example_cluster_ids3.append(tuple(example_cluster_ids_fold))



# handle the case where there is only one template
for train_mask_id, train_mask in enumerate(p_train_masks):
    template = np.clip(
        np.mean(p_padded[train_mask],axis=0),
        .05,.95)
    template = template.reshape(1,p_padded.shape[0],
                                p.padded.shape[1])
    k=1
    save_str = ('/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/080212/'
                + str(k)+'_'+str(train_mask_id)+'_')
    means_vec = np.array([p_lengths[train_mask_id].mean(0)])
    medians_vec = np.array([np.median(p_lengths[train_mask_id],axis=0)])
    np.save(save_str+'p_medians_vec',medians_vec)
    np.save(save_str+'p_means_vec',mean_vec)
    np.save(save_str+'p_templates',template)
    

del bm


for k in [2,3,4,5]:
    print k
    for train_mask_id , train_mask in enumerate(p_train_masks):
        print train_mask_id
        bm
        bm = bernoulli_mixture.BernoulliMixture(k,p_padded[train_mask])
        bm.run_EM(.00001)
        save_str = ('/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/080212/'
                + str(k)+'_'+str(train_mask_id)+'_')
        np.save(save_str+'p_templates',bm.templates           )
        
        

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

np.save(exp_path+"p_padded_templates2.npy",bm2.templates)


bm3 = bernoulli_mixture.BernoulliMixture(3,p_padded)
bm3.run_EM(.00001)

np.save(exp_path+"p_padded_templates3.npy",bm3.templates)

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

"""
Want to see if there is any correlation with context

"""

p3_affinities = bm3.affinities
p3_templates = bm3.templates




bm4 = bernoulli_mixture.BernoulliMixture(4,p_padded)
bm4.run_EM(.00001)

for i in xrange(4):
    print p_lengths[bm4.affinities[:,i] ==1].mean()


np.save(exp_path+"p_padded_templates4.npy",bm4.templates)

    
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
