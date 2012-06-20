import numpy as np
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
#root_path = '/home/mark/projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt
import template_speech_rec.classification as cl

edge_orientations = np.load(root_path+'Experiments/050812/edge_orientations.npy')
abst_threshold = np.load(root_path+'Experiments/050812/abst_threshold.npy')

texp = template_exp.\
    Experiment(patterns=[np.array(('aa','r')),np.array(('ah','r'))],
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=abst_threshold,

               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7,
               offset=3
               )
train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)

train_data_iter.next()
E, edge_feature_row_breaks, edge_orientations =\
    train_data_iter.E,train_data_iter.edge_feature_row_breaks, train_data_iter.edge_orientations


import template_speech_rec.extract_local_features as elf
import matplotlib.pyplot as plt


patch_height,patch_width = (5,5)
lower_quantile = .9
upper_quantile = 1.

bps, spec_patches, signal_windows = elf.local_features_blocks(train_data_iter,30,
                      patch_height,patch_width,
                      lower_quantile,upper_quantile,
                      num_edge_features=8,segment_ms=500,
                      hop_ms = 5)

np.save(root_path+"Experiments/061912/bps061912.npy",bps)
np.save(root_path+"Experiments/061912/spec_patches061912.npy",spec_patches)
np.save(root_path+"Experiments/061912/signal_windows061912.npy",signal_windows)


np.save(root_path+"Experiments/061912/bps_small061912.npy",bps[:1000])
np.save(root_path+"Experiments/061912/spec_patches_small061912.npy",spec_patches[:1000])
np.save(root_path+"Experiments/061912/signal_windows_small061912.npy",signal_windows[:1000])


from matplotlib import cm

#visualize the templates
import matplotlib.pyplot as plt
for n in xrange(bm.templates.shape[0]):
    plt.subplot(4,5,n+1)
    plt.imshow(bm.templates[n],
               interpolation='nearest',cmap = cm.bone)
                                

plt.show()

def patch2template(patch):
    probs = np.minimum(np.maximum(patch,.05),.95)
    return (np.log(probs), np.log(1-probs))

def patches2template(patches):
    return patch2template(np.mean(patches,axis=0))



templates = []
templates.append(patch2template(bp[0]))

def matchtemplate2patch(template,bp):
    t = np.tile(template[0],(bp.shape[0],1,1))
    t_inv = np.tile(template[1],(bp.shape[0],1,1))
    return np.sum(np.sum(t*bp,axis=1),axis=1) + np.sum(np.sum(t_inv *(1-bp),axis=1),axis=1)

scores = matchtemplate2patch(templates[0],bp)
score_idx = np.argsort(scores)
use_patches = bp[score_idx[.97*scores.shape[0]:]]

templates[0] = patches2template(use_patches)

scores = matchtemplate2patch(templates[0],bp)
score_idx = np.argsort(scores)
use_patches = bp[score_idx[.97*scores.shape[0]:]]

from matplotlib import cm

#visualize the templates
import matplotlib.pyplot as plt
for n in xrange(bm.templates.shape[0]):
    plt.subplot(4,5,n+1)
    plt.imshow(patch_templates[0][n],
               interpolation='nearest',cmap = cm.bone)
                                

plt.show()


#
# Going to extract local features again, this time with the different patches tied to each other
#
#

bp = np.zeros((0,edge_orientations.shape[0]*patch_height,patch_width))
train_data_iter.reset_exp()
num_iter = 30
for k in xrange(num_iter):
    train_data_iter.next()
    E, edge_feature_row_breaks, edge_orientations =\
        train_data_iter.E,train_data_iter.edge_feature_row_breaks, train_data_iter.edge_orientations
    esp._edge_map_threshold_segments(E,
                                 40,
                                 1, 
                                 threshold=.3,
                                 edge_orientations = edge_orientations,
                                 edge_feature_row_breaks = edge_feature_row_breaks)
    bp = np.vstack((bp,extract_local_features_tied(E,patch_height,patch_width,.9,1.,edge_feature_row_breaks)))


import template_speech_rec.bernoulli_em as bem



patch_mix80 = bem.Bernoulli_Mixture(80,bps); patch_mix80.run_EM(.00001); np.save('patch_mix80_templates061412',patch_mix80.templates)

out = open('patch_mix80_061912.pkl','wb')
cPickle.dump(patch_mix80,out)
out.close()
del patch_mix80

patch_mix40 = bem.Bernoulli_Mixture(40,bps); patch_mix40.run_EM(.00001); np.save('patch_mix40_templates061412',patch_mix40.templates)

out = open('patch_mix40_061912.pkl','wb')
cPickle.dump(patch_mix40,out)
out.close()
del patch_mix40

patch_mix20 = bem.Bernoulli_Mixture(20,bps); patch_mix20.run_EM(.00001); np.save('patch_mix20_templates061412',patch_mix20.templates)

out = open('patch_mix20_061912.pkl','wb')
cPickle.dump(patch_mix20,out)
out.close()
del patch_mix20

patch_mix20_small = bem.Bernoulli_Mixture(20,bps[:1000]); patch_mix20_small.run_EM(.00001); np.save('patch_mix20_small_templates061912',patch_mix20_small.templates)

out = open('patch_mix20_small_061912.pkl','wb')
cPickle.dump(patch_mix20_small,out)
out.close()
del patch_mix20_small


bp = np.zeros((0,edge_orientations.shape[0]*patch_height,patch_width))
train_data_iter.reset_exp()
num_iter = 30
for k in xrange(num_iter):
    train_data_iter.next()
    E, edge_feature_row_breaks, edge_orientations =\
        train_data_iter.E,train_data_iter.edge_feature_row_breaks, train_data_iter.edge_orientations
    esp._edge_map_threshold_segments(E,
                                 40,
                                 1, 
                                 threshold=.3,
                                 edge_orientations = edge_orientations,
                                 edge_feature_row_breaks = edge_feature_row_breaks)
    bp = np.vstack((bp,extract_local_features_tied(E,patch_height,patch_width,.9,1.,edge_feature_row_breaks)))


patch_mix80_9full= bem.Bernoulli_Mixture(80,bp); patch_mix80_9full.run_EM(.00001); np.save('patch_mix80_9full_templates061412',patch_mix80_9full.templates)

out = open('patch_mix80_9full_061412.pkl','wb')
cPickle.dump(patch_mix80_9full,out)
out.close()
del patch_mix80_9full
