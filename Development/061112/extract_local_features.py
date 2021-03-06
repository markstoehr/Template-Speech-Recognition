import numpy as np
root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'
root_path = '/home/mark/projects/Template-Speech-Recognition/'

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
               #data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               data_paths_file=root_path+'Data/WavFilesTrainPaths',
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

import matplotlib.pyplot as plt





def extract_local_features(E,patch_height,patch_width,lower_quantile,upper_quantile,edge_feature_row_breaks,segment_ms=500,
                           hop_ms = 5):
    # segment_ms - number of milliseconds over which we compute the quantile thresholds
    # hop_ms is says how many milliseconds pass in between each frame
    segment_length = segment_ms/hop_ms
    bps = [np.zeros((0,patch_height,patch_width))]*(edge_feature_row_breaks.shape[0]-1)
    for segment_id in xrange(E.shape[1]/segment_length-1):
        for edge_id in xrange(edge_feature_row_breaks.shape[0]-1):
            bps[edge_id] = np.vstack((bps[edge_id],
                            _extract_block_local_features(
                        E[edge_feature_row_breaks[edge_id]:
                              edge_feature_row_breaks[edge_id+1],
                          segment_id*segment_length:(segment_id+1)*segment_length],
                        patch_height,patch_width,lower_quantile,upper_quantile)))
    return bps

def _extract_block_local_features(E,patch_height,patch_width,lower_quantile,upper_quantile):
    height, width = E.shape
    col_indices = np.tile(
        # construct the base set of repeating column indices
        np.arange(patch_width*(width-patch_width+1)).reshape(width-patch_width+1,1,patch_width,),
        # repeat the same indices (for columns or times) along each row as those are fixed
        (1,patch_height,1))
    # change the entries so that way the col[i,:,:] starts with i along the first column
    col_indices = col_indices/patch_width + col_indices%patch_width
    # repeat for each set of frequency bands
    col_indices = np.tile(col_indices,(height-patch_height+1,1,1))
    #
    # construct the row indices
    #
    #
    # get the base indices
    row_indices = np.tile(np.arange(0,width*patch_height,width),(patch_width,1)).T
    # tile them so that way we have as many as are there are of the col_indices
    row_indices = np.tile(
        row_indices.reshape(1,patch_height,patch_width),
        (col_indices.shape[0],1,1)
        )
    #
    # now we make the mask that we do our shifting with
    row_add_mask = (np.arange(col_indices.size,dtype=int)/(patch_height*patch_width*(width-patch_width+1))).reshape(col_indices.shape)
    row_add_mask *= width
    row_indices += row_add_mask
    patches = E.ravel()[row_indices+col_indices]
    return patches[np.argsort(np.sum(np.sum(patches,axis=1),axis=1))[lower_quantile*patches.shape[0]:
                                                                         upper_quantile*patches.shape[0]]]


def extract_local_features_spec(S,patch_height,patch_width,
                                quantile,data_iter,segment_ms = 500,hop_ms=5):
    """
    Version 2 for the local feature extraction function
    """
    # S is a spectrogram
    # segment_ms - number of milliseconds over which we compute the quantile thresholds
    # hop_ms is says how many milliseconds pass in between each frame
    data_iter.next(compute_S=True)
    S = data_iter.S
    E, edge_feature_row_breaks, edge_orientations =\
        data_iter.E,data_iter.edge_feature_row_breaks, data_iter.edge_orientations
    bp = extract_local_features_tied(E,patch_height,patch_width,
                                     quantile,1.,
                                     edge_feature_row_breaks,segment_ms=segment_ms,
                                     hop_ms = hop_ms)
    


def extract_local_features_tied(E,patch_height,patch_width,
                                lower_quantile,upper_quantile,
                                edge_feature_row_breaks,segment_ms=500,
                           hop_ms = 5):
    """
    Version 3 of the local feature extraction code, this time we can associate row indices
    and column indices with the extracted patches:

    Typical Usage is:
    bp,all_patch_row,all_patch_cols = extract_local_features_tied(E,patch_height,
                                                                  patch_width, lower_quantile,
                                                                  upper_quantile, edge_feature_row_breaks)

    """
    # segment_ms - number of milliseconds over which we compute the quantile thresholds
    # hop_ms is says how many milliseconds pass in between each frame
    segment_length = segment_ms/hop_ms
    bp = np.zeros((0,patch_height*(edge_feature_row_breaks.shape[0]-1),
                   patch_width))
    # keeps track of which patch is associated with what row and column of E, and in turn, the spectrogram
    # that generated E
    all_patch_rows = np.zeros(0)
    all_patch_cols = np.zeros(0)
    for segment_id in xrange(E.shape[1]/segment_length-1):
        patch_row_ids, patch_col_ids = get_flat_patches2E_indices(E[edge_feature_row_breaks[0]:
                                                                  edge_feature_row_breaks[1],
                                                              segment_id*segment_length:
                                                                  (segment_id+1)*segment_length],
                                                            patch_height,patch_width)
        patch_col_ids += segment_id*segment_length
        bp_tmp = _extract_block_local_features_tied(
                        E[edge_feature_row_breaks[0]:
                              edge_feature_row_breaks[1],
                          segment_id*segment_length:(segment_id+1)*segment_length],
                        patch_height,patch_width)
        for edge_id in xrange(1,edge_feature_row_breaks.shape[0]-1):
            bp_tmp = np.hstack((bp_tmp,
                            _extract_block_local_features_tied(
                        E[edge_feature_row_breaks[edge_id]:
                              edge_feature_row_breaks[edge_id+1],
                          segment_id*segment_length:(segment_id+1)*segment_length],
                        patch_height,patch_width)))
        use_indices = np.argsort(np.sum(np.sum(bp_tmp,axis=1),axis=1))[lower_quantile*bp_tmp.shape[0]:
                                                                           upper_quantile*bp_tmp.shape[0]]
        all_patch_rows = np.hstack((all_patch_rows,patch_row_ids[use_indices]))
        all_patch_cols = np.hstack((all_patch_cols,patch_col_ids[use_indices]))
        bp_tmp=bp_tmp[use_indices]
        bp = np.vstack((bp,bp_tmp))
    return bp,all_patch_rows,all_patch_cols

def get_flat_patches2E_indices(E,patch_height,patch_width):
    """
    Patches are just in a matrix, where the first index indexes the patch order
    they are grabbed from E in row-column order.
    
    So the patch dimensionality is (num_patches,patch_height,patch_width)

    For each such index this function
    produces two arrays that map the patch index to the row of E its associated with
    and to the column of E
    """
    num_patches_across = E.shape[1] - patch_width+1
    num_patches_down = E.shape[0] - patch_height+1
    num_patches = num_patches_across * num_patches_down
    patch_row_ids = np.arange(num_patches) / num_patches_across
    patch_col_ids = np.arange(num_patches) % num_patches_across
    return patch_row_ids, patch_col_ids

def _extract_block_local_features_tied(E,patch_height,patch_width):
    height, width = E.shape
    col_indices = np.tile(
        # construct the base set of repeating column indices
        np.arange(patch_width*(width-patch_width+1)).reshape(width-patch_width+1,1,patch_width,),
        # repeat the same indices (for columns or times) along each row as those are fixed
        (1,patch_height,1))
    # change the entries so that way the col[i,:,:] starts with i along the first column
    col_indices = col_indices/patch_width + col_indices%patch_width
    # repeat for each set of frequency bands
    col_indices = np.tile(col_indices,(height-patch_height+1,1,1))
    #
    # construct the row indices
    #
    #
    # get the base indices
    row_indices = np.tile(np.arange(0,width*patch_height,width),(patch_width,1)).T
    # tile them so that way we have as many as are there are of the col_indices
    row_indices = np.tile(
        row_indices.reshape(1,patch_height,patch_width),
        (col_indices.shape[0],1,1)
        )
    #
    # now we make the mask that we do our shifting with
    row_add_mask = (np.arange(col_indices.size,dtype=int)/(patch_height*patch_width*(width-patch_width+1))).reshape(col_indices.shape)
    row_add_mask *= width
    row_indices += row_add_mask
    patches = E.ravel()[row_indices+col_indices]
    return patches


patch_height,patch_width = 5,5
bp = extract_local_features_tied(E,patch_height,patch_width,.85,.95,edge_feature_row_breaks)

assert np.sum(np.abs(patches[best_patches] - bp)) == 0
#
#
# we will try an experiment with spreading and without spreading
#



# so we have now tested the local feature extraction process
# time to now extract the local features from a large set of utterances
#

# Going to create a matrix that will have all the patches we want
# we have (width-5) * (height-5) patches so the patch matrix will have 
# 5 *(width-5) columns

# bp = extract_local_features(E,patch_height,patch_width,.8,.95)

patch_height, patch_width = 5,5

bps = [np.zeros((0,patch_height,patch_width))] * 8
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
    cur_bps = extract_local_features(E,patch_height,patch_width,.85,.95,edge_feature_row_breaks)
    for bp_id in xrange(len(cur_bps)):
        bps[bp_id] = np.vstack((bps[bp_id],
                                cur_bps[bp_id]))


import template_speech_rec.bernoulli_em as bem


patch_mixes = [bem.Bernoulli_Mixture(20,bps[k]) for k in xrange(8)]
for k in xrange(8):
    patch_mixes[k].run_EM(.0001)

out = open(root_path + 'Experiments/061112/patch_mixes.pkl','wb')
cPickle.dump(patch_mixes,out)
out.close()

patch_templates = [patch_mixes[k].templates for k in xrange(8)]

out = open(root_path + 'Experiments/061112/patch_templates.pkl','wb')
cPickle.dump(patch_templates,out)
out.close()
 

np.save('patch_data_mat061211',bm.data_mat)
bm.data_mat = 0

pkl_out = open('patch_templates061211.pkl','wb')
cPickle.dump(bm,pkl_out)
pkl_out.close()


pkl_out = open('patch_templates061211.pkl','rb')
bm = cPickle.load(pkl_out)
pkl_out.close()


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


patch_mix4 = bem.Bernoulli_Mixture(4,bp); patch_mix4.run_EM(.1); np.save('patch_mix4_templates061412',patch_mix4.templates)

patch_mix20 = bem.Bernoulli_Mixture(20,bp); patch_mix20.run_EM(.00001); np.save('patch_mix20_templates061412',patch_mix20.templates)
patch_mix40 = bem.Bernoulli_Mixture(40,bp); patch_mix40.run_EM(.00001); np.save('patch_mix40_templates061412',patch_mix40.templates)
patch_mix80 = bem.Bernoulli_Mixture(80,bp); patch_mix80.run_EM(.00001); np.save('patch_mix80_templates061412',patch_mix80.templates)

out = open('patch_mix80_061412.pkl','wb')
cPickle.dump(patch_mix80,out)
out.close()
del patch_mix80


out = open('patch_mix40_061412.pkl','wb')
cPickle.dump(patch_mix40,out)
out.close()
del patch_mix40


out = open('patch_mix20_061412.pkl','wb')
cPickle.dump(patch_mix20,out)
out.close()
del patch_mix20

out = open('patch_mix20_061412.pkl','rb')
patch_mix20=cPickle.load(out)
out.close()

del patch_mix20


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
