import numpy as np
from collections import defaultdict
import collections
#root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
root_path = '/home/mark/Template-Speech-Recognition/'
data_path = root_path + 'Data/Train/'
tmp_data_path = root_path+'Experiments/080612/data/'
old_exp_path = root_path+'Experiments/080312/'
model_save_dir = root_path+'Experiments/080612/models/'
import sys, os, cPickle
sys.path.append(root_path)

import template_speech_rec.test_template as tt
import template_speech_rec.bernoulli_mixture as bernoulli_mixture

phns = ['p','t','k','b','d','g','iy','aa','uw']
num_mixtures_set = [2,4,6,8,10,12,14]


BernoulliModelsByPhone = collections.namedtuple(
    'BernoulliModelsByPhone',
    ' '.join(phns))
bmodels = []
for phn in phns:
    phn_examples5 = np.swapaxes(np.load(data_path+phn+'class_examples5.npy'),1,2)
    phn_lengths = np.load(data_path+phn+'class_examples_lengths.npy')
    phn_bgs = np.load(data_path+phn+'class_examples_bgs.npy')
    phn_train_masks = np.load(old_exp_path+phn+'_train_masks.npy')
    phnModels = []
    for mix_id, num_mixtures in enumerate(num_mixtures_set):
        mixModels = []
        for mask_id, train_mask in enumerate(phn_train_masks):
            bm = bernoulli_mixture.BernoulliMixture(num_mixtures,phn_examples5[train_mask])
            bm.run_EM(.000001)
            cluster_lists = bm.get_cluster_lists()
            template_lengths = tuple(int(phn_lengths[cluster_list].mean()+.5)\
                                         for cluster_list in cluster_lists)
            bms = bernoulli_mixture.BernoulliMixtureSimple(
                log_templates=tuple(log_template[:l] for log_template,l in zip(bm.log_templates,
                                                                               template_lengths)),
                log_invtemplates=tuple(log_invtemplate[:l] for log_invtemplate,l in zip(bm.log_invtemplates,
                                                                                        template_lengths)),
                weights=bm.weights)
            out = open(model_save_dir+phn+'_'+str(num_mixtures)+'_'+str(mask_id)+'bms','wb')
            cPickle.dump(bms,out)
            out.close()
            mixModels.append(bms)
        phnModels.append(tuple(mixModels))
    bmodels.append(tuple(phnModels))

bmodels = tuple(bmodels)
        
        
p_lengths = np.load(exp_load_path+phn+'class_examples_lengths.npy')
p_bgs = np.load(exp_load_path+phn+'class_examples_bgs.npy')

p_utt_id_E_loc = np.load(exp_load_path+phn+'example_utt_id_E_loc.npy')
p_specs = np.zeros((p_bgs.shape[0],
                    p_bgs.shape[1]/8,
                    p_examples5.shape[2]),dtype=np.float32)

p_examples3 = np.load(exp_load_path+phn+'class_examples3.npy')


b_examples5 = np.load(exp_load_path+'bclass_examples5.npy')
b_lengths = np.load(exp_load_path+'bclass_examples_lengths.npy')
b_bgs = np.load(exp_load_path+'bclass_examples_bgs.npy')
b_train_masks = np.load(exp_load_path+'b_train_masks.npy')
b_utt_id_E_loc = np.load(exp_load_path+'bexample_utt_id_E_loc.npy')
b_specs = np.zeros((b_bgs.shape[0],
                    b_bgs.shape[1]/8,
                    b_examples5.shape[2]),dtype=np.float32)


b_examples3 = np.load(exp_load_path+'bclass_examples3.npy')

def make_padded_example(example,length,bg):
    residual_length = example.shape[1] - length
    height = len(bg)
    if residual_length > 0:
        return np.hstack(
            (example[:,:length],
             (np.random.rand(height,residual_length) < \
                  np.tile(bg,(residual_length,
                              1)).T).astype(np.uint8)))
    return example


         
def make_padded(examples,lengths,bgs):
    return np.array([
            make_padded_example(example,length,bg)\
            for example,length, bg in \
            zip(examples,lengths,bgs)])

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
                                                     


b_padded = make_padded(b_examples5,
                       b_lengths,
                       b_bgs)


out = open(exp_load_path+'example_cluster_ids3.pkl','rb')
example_cluster_ids3 = cPickle.load(out)
out.close()

out = open(exp_load_path+'example_cluster_ids4.pkl','rb')
example_cluster_ids4 = cPickle.load(out)
out.close()

p_example_cluster_ids = example_cluster_ids3 + example_cluster_ids4
p_cluster_nums = tuple((len(ci[0]) for ci in p_example_cluster_ids))

p_train_masks = np.load(exp_load_path+'p_train_masks.npy')
b_train_masks = np.load(exp_load_path+'b_train_masks.npy')

b_dev_likelihoods = np.zeros((len(p_cluster_nums),len(p_train_masks)))
b_example_cluster_ids = []
for mix_id, k in enumerate(p_cluster_nums):
    example_cluster_ids_fold = []
    print k
    for train_mask_id , train_mask in enumerate(b_train_masks):
        medians_vec = np.zeros(k)
        mean_vec = np.zeros(k)
        spec_avgs = np.zeros((k,b_specs.shape[1], b_specs.shape[2]),dtype=np.float32)
        spec_meds = np.zeros((k,b_specs.shape[1], b_specs.shape[2]),dtype=np.float32)
        E_avgs = np.zeros((k,b_examples5.shape[1], b_examples5.shape[2]),dtype=np.float32)
        E_meds = np.zeros((k,b_examples5.shape[1], b_examples5.shape[2]),dtype=np.float32)
        print train_mask_id
        bm = bernoulli_mixture.BernoulliMixture(k,b_padded[train_mask])
        bm.run_EM(.00001)
        bernoulli_model = BernoulliMixtureSimple(
            log_templates=bm.log_templates,
            log_invtemplates=bm.log_invtemplates,
            weights=bm.weights,
            num_mix=k)
        b_dev_likelihoods[mix_id,train_mask_id] = bernoulli_model_loglike(bernoulli_model,b_padded[True-train_mask],
                                                                          use_weights=True)
        cluster_ids = []
        for i in xrange(k):
            cluster_ids.append(bm.affinities[:,i] > .99)
            mix_lengths = b_lengths[bm.affinities[:,i] >.99]
            medians_vec[i] = np.median(mix_lengths)
            mean_vec[i] = np.mean(mix_lengths)
            spec_avgs[i] = b_specs[bm.affinities[:,i] > .99].mean(0)
            spec_meds[i] = np.median(b_specs[bm.affinities[:,i] > .99],axis=0)
            E_avgs[i] = np.mean(b_examples5[bm.affinities[:,i] > .99],axis=0)
            E_meds[i] = np.median(b_examples5[bm.affinities[:,i] > .99],axis=0)
        example_cluster_ids_fold.append(tuple(cluster_ids))
        save_str = ('/home/mark/Template-Speech-Recognition/Experiments/080212/'
                + str(k)+'_'+str(train_mask_id)+'_')
        np.save(save_str+'b_medians_vec',medians_vec)
        np.save(save_str+'b_means_vec',mean_vec)
        np.save(save_str+'b_spec_avgs',spec_avgs)
        np.save(save_str+'b_spec_meds',spec_meds)
        np.save(save_str+'b_E_avgs',E_avgs)
        np.save(save_str+'b_E_meds',E_meds)
        np.save(save_str+'b_templates',bm.templates           )
        np.save(save_str+'b_affinites',bm.affinities           )
    b_example_cluster_ids.append(tuple(example_cluster_ids_fold))


out = open('b_example_cluster_ids.pkl','wb')
cPickle.dump(b_example_cluster_ids,out)
out.close()


classify_pbtable = collections.namedtuple('classify_pbtable',
                                        'p_classify_p b_classify_p p_classify_b b_classify_b')


p_error_rates = np.zeros((len(p_example_cluster_ids),len(b_example_cluster_ids),
                          10))
b_error_rates = np.zeros((len(p_example_cluster_ids),len(b_example_cluster_ids),
                          10))
for p_cluster_id, p_cluster_set in enumerate(p_example_cluster_ids):
    print 'p',p_cluster_id,len(p_cluster_set[0])
    for b_cluster_id, b_cluster_set in enumerate(b_example_cluster_ids):
        print 'b',b_cluster_id,len(b_cluster_set[0])
        p_error_rates_cluster = []
        b_error_rates_cluster = []
        for train_mask_id in xrange(10):
            print train_mask_id
            p_num_components = len(p_cluster_set[train_mask_id])
            b_num_components = len(b_cluster_set[train_mask_id])
            # get the p_components
            p_components = []
            b_components = []
            for component_id in xrange(p_num_components):
                p_component_length = (int(p_lengths[p_train_masks[train_mask_id]][p_cluster_set[train_mask_id][component_id]].mean()+.5))
                p_components.append(
                    np.clip(p_examples5[p_train_masks[train_mask_id]][p_cluster_set[train_mask_id][component_id]].mean(0)[:,:p_component_length],.05,.95))
            for component_id in xrange(b_num_components):
                b_component_length = (int(b_lengths[b_train_masks[train_mask_id]][b_cluster_set[train_mask_id][component_id]].mean()+.5))
                b_components.append(
                    np.clip(b_examples5[b_train_masks[train_mask_id]][b_cluster_set[train_mask_id][component_id]].mean(0)[:,
                            :b_component_length],.05,.95))
            p_components = tuple(p_components)
            b_components = tuple(b_components)
            p_dev_exs = p_examples3[True-p_train_masks[train_mask_id]]
            p_dev_bgs = p_bgs[True-p_train_masks[train_mask_id]]
            p_classify_p = np.array([
                    max((sum(tt.score_template_background_section(p_comp,p_dev_bg,p_dev_ex[:,:p_comp.shape[1]])) for p_comp in p_components)) for p_dev_bg, p_dev_ex in zip(p_dev_bgs, p_dev_exs)])
            b_classify_p = np.array([
                    max((sum(tt.score_template_background_section(b_comp,p_dev_bg,p_dev_ex[:,:b_comp.shape[1]])) for b_comp in b_components)) for p_dev_bg, p_dev_ex in zip(p_dev_bgs, p_dev_exs)])
            b_dev_exs = b_examples3[b_train_masks[train_mask_id]][True-b_cluster_set[train_mask_id][component_id]]
            b_dev_bgs = b_bgs[b_train_masks[train_mask_id]][True-b_cluster_set[train_mask_id][component_id]]
            p_classify_b = np.array([
                    max((sum(tt.score_template_background_section(p_comp,b_dev_bg,b_dev_ex[:,:p_comp.shape[1]])) for p_comp in p_components)) for b_dev_bg, b_dev_ex in zip(b_dev_bgs, b_dev_exs)])
            b_classify_b = np.array([
                    max((sum(tt.score_template_background_section(b_comp,b_dev_bg,b_dev_ex[:,:b_comp.shape[1]])) for b_comp in b_components)) for b_dev_bg, b_dev_ex in zip(b_dev_bgs, b_dev_exs)])
            p_errors = np.sum(p_classify_p < b_classify_p)/float(p_classify_p.shape[0])
            b_errors = np.sum(p_classify_b > b_classify_b)/float(b_classify_b.shape[0])
            p_error_rates[p_cluster_id,b_cluster_id,
                          train_mask_id] = p_errors
            b_error_rates[p_cluster_id,b_cluster_id,
                          train_mask_id] = b_errors
            print "p_errors:",p_errors
            print "b_errors:",b_errors

pb_prod_errors = np.zeros(p_error_rates.shape)
for i in xrange(pb_prod_errors.shape[0]):
    for j in xrange(pb_prod_errors.shape[1]):
        pb_prod_errors[i,j] = p_error_rates[i,j] * b_error_rates[i,j]

pb_sum_errors = p_error_rates + b_error_rates
np.save(exp_load_path+'p_error_rates',p_error_rates)
np.save(exp_load_path+'b_error_rates',b_error_rates)


#
# want to understand the confusion between these two
# increase mixtures in component until we do better
#
# use mixtures of the different lengths and identities, maybe use the resizing
# or don't use the resizing instead just use the average of the padded backgrounds
# and use the average lengths




out = open(exp_load_path+'b_example_cluster_ids.pkl','rb')
b_example_cluster_ids = cPickle.load(out)
out.close()

#
#
# want to do a check of consistency over the different folds
# count how many are active across folds, in particular check
# whether all the guys in a given affinity cluster to the same mixture 
# component for other mixtures

# write the affinity checking mechanism


# first step is to repeat everything that was done for /p/ and do it for /b/
# want to do a comparison of how the different number of clusters compete
# against each other

# collect the cluster numbers

p_cluster_nums = tuple((len(ci[0]) for ci in p_example_cluster_ids))
