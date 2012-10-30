import numpy as np
import template_speech_rec.get_train_data as gtrd
import template_speech_rec.estimate_template as et
import template_speech_rec.bernoulli_mixture as bm
import cPickle

sp = gtrd.SpectrogramParameters(
    sample_rate=16000,
    num_window_samples=320,
    num_window_step_samples=80,
    fft_length=512,
    kernel_length=7,
    freq_cutoff=3000,
    use_mel=False)

ep = gtrd.EdgemapParameters(block_length=40,
                            spread_length=1,
                            threshold=.7)

utterances_path = '/home/mark/Template-Speech-Recognition/Data/Train/'
file_indices = gtrd.get_data_files_indices(utterances_path)

syllable=('aa','r')
syllable_features,avg_bgd=gtrd.get_syllable_features_directory(utterances_path,file_indices,syllable,
                                    S_config=sp,E_config=ep,offset=0,
                                    E_verbose=False,return_avg_bgd=True)

np.save('data/aar_bgd.npy',avg_bgd.E)

example_mat = gtrd.recover_example_map(syllable_features)
lengths,waveforms  = gtrd.recover_waveforms(syllable_features,example_mat)

np.savez('data/aar_waveforms_lengths.npz',waveforms,lengths,example_mat)

Slengths,Ss  = gtrd.recover_specs(syllable_features,example_mat)

np.savez('data/aar_Ss_lengths.npz',Ss,Slengths,example_mat)


lengths,Es  = gtrd.recover_edgemaps(syllable_features,example_mat)

np.savez('data/aar_Es_lengths.npz',Es,lengths,example_mat)

#
# Now that we have the Es we estimate the model
#
bgd = np.clip(avg_bgd.E,.1,.4)
for i in xrange(Es.shape[0]):
    if lengths[i] < Es.shape[1]:
        Es[i][lengths[i]:] = np.tile(bgd,(Es.shape[1]-lengths[i],1,1))
np.save('data/aar1_padded_examplesE.npy',Es)



num_mix = 2
import matplotlib.pyplot as plt

num_mix_params = [2,3,4,5,6,7,8,9]
for num_mix in num_mix_params:
    print num_mix
    bem = bm.BernoulliMixture(num_mix,Es)
    bem.run_EM(.000001)
    templates = et.recover_different_length_templates(bem.affinities,
                                                        Es,
                                                        lengths)
    spec_templates = et.recover_different_length_templates(bem.affinities,
                                                           Ss,
                                                           Slengths)
    num_plots = len(spec_templates)
    num_rows = 2
    num_cols = num_plots/num_rows+1
    for i in xrange(len(spec_templates)):
        plt.subplot(num_cols,num_rows,i+1)
        plt.imshow(spec_templates[i].T[::-1],interpolation='nearest')
    plt.savefig('aar1_spec_templates_%d.png' % num_mix)
    np.savez('aar1_templates_%d.npz' % num_mix, templates)
    np.save('aar1_affinities_%d.npy' % num_mix, bem.affinities)


test_path = '/home/mark/Template-Speech-Recognition/Data/Test/'
test_file_indices = gtrd.get_data_files_indices(test_path)

test_example_lengths = gtrd.get_detect_lengths(test_file_indices,test_path)
np.save("data/test_example_lengths.npy",test_example_lengths)
np.save("data/test_file_indices.npy",test_file_indices)


for num_mix in num_mix_params:
    templates = (np.load('aar1_templates_%d.npz' % num_mix))['arr_0']
    detection_array = np.zeros((test_example_lengths.shape[0],
                            test_example_lengths.max() + 2),dtype=np.float32)
    linear_filters_cs = et.construct_linear_filters(templates,
                                                    bgd)
    np.savez('data/linear_filter_aar_%d.npy'% num_mix,linear_filters_cs[:][0])
    np.savez('data/c_aar_%d.npy'%num_mix,np.array(linear_filters_cs[:][1]))
    syllable = np.array(['aa','r'])
    detection_array,example_start_end_times, detection_lengths = gtrd.get_detection_scores_mixture(test_path,
                                                                                                   detection_array,
                                                                                                   syllable,
                                                                                                   linear_filters_cs,
                                                                                                                                                                               verbose=True)
    np.save('data/detection_array_aar_%d.npy' % num_mix,detection_array)
    if num_mix == 2:
        out = open('data/example_start_end_times_aar.pkl','wb')
        cPickle.dump(parts_example_start_end_times,out)
        out.close()
        out = open('data/detection_lengths_aar.pkl','wb')
        cPickle.dump(parts_detection_lengths,out)
        out.close()
    window_start = -10
    window_end = 10
    max_detect_vals = rf.get_max_detection_in_syllable_windows(detection_array,
                                                        example_start_end_times,
                                                               detection_lengths,
                                                        window_start,
                                                        window_end)
    np.save('data/max_detect_vals_aar_%d.npy' % num_mix,max_detect_vals)
    C0 = 33
    C1 = int( 33 * 1.5 + .5)
    frame_rate = 1/.005
    fpr, tpr = rf.get_roc_curve(max_detect_vals,
                                detection_array,
                                np.array(detection_lengths),
                        example_start_end_times,
                        C0,C1,frame_rate)
    np.save('data/fpr_aar_%d.npy' % num_mix,
            fpr)
    np.save('data/tpr_aar_%d.npy' % num_mix,
            tpr)
    detection_clusters = rf.get_detect_clusters_threshold_array(max_detect_vals,
                                                                detection_array,
                                                                np.array(detection_lengths),
                                                                C0,C1)
    out = open('data/detection_clusters_aar_%d.npy' % num_mix,
               'wb')
    cPickle.dump(detection_clusters,out)
    out.close()
    for i in xrange(1,11):
        thresh_idx = np.arange(fpr.shape[0])[fpr*60 <= i].min()
        pos_cluster_responses,neg_cluster_responses = get_pos_neg_detections(detection_clusters[thresh_idx],detection_array,C1,window_start,window_end,example_start_end_times)
        if pos_cluster_responses.shape[0] > 0:
            np.save("data/aar_pos_cluster_responses_%d_%d.npy"% (num_mix,i),pos_cluster_responses)
            pos_response_grid, pos_response_points = rf.map_cluster_responses_to_grid(pos_cluster_responses)
            rf.display_response_grid("aar_pos_response_grid_%d_%d.png" % (num_mix,i),pos_response_grid,pos_response_points)
        if neg_cluster_responses.shape[0] > 0:
            np.save("data/aar_neg_cluster_responses_%d_%d.npy"%(num_mix,i),neg_cluster_responses)
            neg_response_grid, neg_response_points = rf.map_cluster_responses_to_grid(neg_cluster_responses)
            rf.display_response_grid("aar_neg_response_grid_%d_%d.png" % (num_mix,i),neg_response_grid,neg_response_points)


import itertools    

def get_threshold_neighborhood(cluster,detection_row,C1):
    start_idx = int((cluster[0]+cluster[1])/2. - C1/2.+.5)
    end_idx = start_idx + C1
    return np.hstack((
            np.zeros(-min(start_idx,0)),
            detection_row[max(start_idx,0):min(end_idx,detection_row.shape[0])],
            np.zeros(max(end_idx-detection_row.shape[0],0))))

    
def get_pos_neg_detections(detection_clusters_at_threshold,detection_array,C1,window_start,window_end,example_start_end_times):
    num_clusters = sum( len(cset) for cset in detection_clusters_at_threshold)
    num_pos_clusters = 0
    num_neg_clusters = 0
    pos_clusters = np.zeros((num_clusters,C1))
    neg_clusters = np.zeros((num_clusters,C1))
    for detect_clusters, detection_row, start_end_times in itertools.izip(detection_clusters_at_threshold,detection_array,example_start_end_times):
        for c in detect_clusters:
            is_neg = True
            for s,e in start_end_times:
                if s-window_start <= c[1] and s+window_end >= c[0]:
                    is_neg = False
                    pos_clusters[num_pos_clusters] = get_threshold_neighborhood(c,detection_row,C1)
                    num_pos_clusters += 1
            if is_neg:
                neg_clusters[num_neg_clusters] = get_threshold_neighborhood(c,detection_row,C1)
                num_neg_clusters += 1
    return pos_clusters[:num_pos_clusters], neg_clusters[:num_neg_clusters]
    
get_pos_neg_detections(detection_clusters[-1],detection_array,C1,window_start,window_end,example_start_end_times)


from scipy.stats import gaussian_kde

def map_cluster_responses_to_grid(cluster_responses):
    cluster_length = cluster_responses.shape[1]
    response_grid = np.zeros((cluster_length,cluster_length))
    response_points = np.arange(cluster_length) * (cluster_responses.max() - cluster_responses.min())/cluster_length + cluster_responses.min()
    for col_idx,response_col in enumerate(cluster_responses.T):
        col_pdf = gaussian_kde(response_col)
        response_grid[col_idx] = col_pdf(response_points)
    return response_grid.T,response_points

def display_response_grid(fname,response_grid,response_points,point_spacing=10):
    plt.close()
    plt.imshow(response_grid[::-1])
    plt.yticks(np.arange(response_points.shape[0])[::10],response_points[::-point_spacing].astype(int))
    plt.savefig(fname)


out = open('data/aar_syllable_features1.pkl','wb')
cPickle.dump(syllable_features,out)
out.close()


get_syllable_features_directory(utterances_path,file_indices,syllable,
                                    S_config=None,E_config=None,offset=None)

