import sys, os
#sys.path.append('/home/mark/projects/Template-Speech-Recognition')
sys.path.append('/var/tmp/stoehr/Projects/Template-Speech-Recognition')

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


sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
freq_cutoff = 3000
exp_path_files_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/042712/'
s_files_path_file = exp_path_files_dir+'et_path_files_s.txt'
phns_files_path_file = exp_path_files_dir+'et_path_files_phns.txt'
phn_times_files_path_file = exp_path_files_dir+'et_path_files_phn_times.txt'

# /home/mark/projects
data_dir = "/home/mark/projects/Template-Speech-Recognition/Data/WavFilesTrain"
#data_dir = "/"
#data_dir = "/var/tmp/stoehr/Projects/edges/WavFilesTrain/"
pattern = np.array(('aa','r'))
kernel_length = 7


exp = t_exp.Experiment(sample_rate,freq_cutoff,
                       num_window_samples,
                       num_window_step_samples,
                       fft_length,s_files_path_file,
                       phns_files_path_file,
                       phn_times_files_path_file,
                       data_dir,pattern,kernel_length)

#
# Get the templates
#
#

class AverageBackground:
    def __init__(self):
        self.num_frames = 0
        self.processed_frames = False
    # Method to add frames
    def add_frames(self,E,edge_feature_row_breaks,
                   edge_orientations,abst_threshold):
        new_E = E.copy()
        esp.threshold_edgemap(new_E,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(new_E,edge_feature_row_breaks,edge_orientations,spread_length=3)
        if not self.processed_frames:
            self.E = np.mean(new_E,axis=1)
            self.processed_frames = True
        else:
            self.E = (self.E * self.num_frames + np.sum(new_E,axis=1))/(self.num_frames+new_E.shape[1])
        self.num_frames += new_E.shape[1]
        

E_avg = AverageBackground()            


all_patterns = []
# these are the fronts and backs of the syllable saved to estimate the template bits
# just randomly these are going to be length 10 with five of the frames from before the syllable start
all_fronts_backs = []
all_bgds = []
bg_len = 26
empty_bgds = []
pattern_num = 0
abst_threshold = np.array([0.025,0.025,0.015,0.015,
                           0.02,0.02,0.02,0.02])

all_raw_patterns_context = []
all_raw_bgds = []


# number of thresholds per data point is 8
pattern_edge_thresholds = []
pattern_lengths = []
bgd_edge_thresholds=[]


mean_template_length = 33

for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if True: #exp.has_pattern(phns):
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # select the object
        patterns = exp.get_patterns(E,phns,phn_times,s)
        patterns_context = exp.get_patterns(E,phns,phn_times,s,context=True,template_length=33)
        bgds = exp.get_pattern_bgds(E,phns,phn_times,s,bg_len)
        fronts_backs = exp.get_pattern_fronts_backs(E,phns,phn_times,s,bg_len)
        E_avg.add_frames(E,edge_feature_row_breaks,
                   edge_orientations,abst_threshold)
        # threshold pattern edges
        for i in xrange(len(patterns)):
            all_raw_patterns_context.append(patterns_context[i].copy())
            all_raw_bgds.append(bgds[i].copy())
            _, edge_thresholds = esp.threshold_edgemap(patterns[i],.30,edge_feature_row_breaks,report_level=True,abst_threshold=abst_threshold)
            # we record both the thresholds
            # and the length to see if there is a relationship
            pattern_edge_thresholds.append(edge_thresholds)
            pattern_lengths.append(patterns[i].shape[1])
            esp.spread_edgemap(patterns[i],edge_feature_row_breaks,edge_orientations,spread_length=5)
            if bgds[i].shape[1] > 0:
                _,edge_thresholds = esp.threshold_edgemap(bgds[i],.30,edge_feature_row_breaks,report_level=True,abst_threshold=abst_threshold)
                bgd_edge_thresholds.append(edge_thresholds)
                esp.spread_edgemap(bgds[i],edge_feature_row_breaks,edge_orientations,spread_length=5)
                # compute background mean
                bgds[i] = np.mean(bgds[i],axis=1)
                # impose floor and ceiling constraints on values
                bgds[i] = np.maximum(np.minimum(bgds[i],.4),.05)
            else:
                bgds[i] = np.random.rand(patterns[i].shape[0]).reshape(patterns[i].shape[0],1)
                bgds[i] = np.mean(bgds[i],axis=1)
                bgds[i] = np.maximum(np.minimum(bgds[i],.4),.05)
                empty_bgds.append(pattern_num)
        pattern_num += len(patterns)
        all_patterns.extend(patterns)
        all_bgds.extend(bgds)


mean_background = E_avg.E.copy()
mean_background = np.maximum(np.minimum(mean_background,.4),.05)


template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns,template_length=33)

template_shape = np.array([template_height,template_length])
np.save('mean_template043012',mean_template)
np.save('template_shape043012',template_shape)


#
# In this section we estimate the value that j0 should have
#
##


sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
freq_cutoff = 3000
exp_path_files_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/042712/'
s_files_path_file = exp_path_files_dir+'j0_path_files_s.txt'
phns_files_path_file = exp_path_files_dir+'j0_path_files_phns.txt'
phn_times_files_path_file = exp_path_files_dir+'j0_path_files_phn_times.txt'

# /home/mark/projects
data_dir = "/home/mark/projects/Template-Speech-Recognition/Data/WavFilesTrain"
#data_dir = "/"
#data_dir = "/var/tmp/stoehr/Projects/edges/WavFilesTrain/"
pattern = np.array(('aa','r'))
kernel_length = 7


exp = t_exp.Experiment(sample_rate,freq_cutoff,
                       num_window_samples,
                       num_window_step_samples,
                       fft_length,s_files_path_file,
                       phns_files_path_file,
                       phn_times_files_path_file,
                       data_dir,pattern,kernel_length)



E_avg = AverageBackground()            


abst_threshold = np.array([0.025,0.025,0.015,0.015,
                           0.02,0.02,0.02,0.02])

all_raw_patterns_context = []

mean_template_length = 33

for path_idx in xrange(exp.num_data):
    if path_idx % 10 ==0:
        print "on path", path_idx
    phns = exp.get_phns(path_idx)
    # check if this datum has what we need
    if exp.has_pattern(phns):
        s = exp.get_s(path_idx)
        E,edge_feature_row_breaks,\
            edge_orientations= exp.get_edgemap_no_threshold(s)
        phn_times = exp.get_phn_times(path_idx)
        # select the object
        patterns_context = exp.get_patterns(E,phns,phn_times,s,context=True,template_length=33)
        E_avg.add_frames(E,edge_feature_row_breaks,
                         edge_orientations,abst_threshold)
        patterns = exp.get_patterns(E,phns,phn_times,s)
        bgds = exp.get_pattern_bgds(E,phns,phn_times,s,bg_len)
        # threshold pattern edges
        for i in xrange(len(patterns)):
            all_raw_patterns_context.append(patterns_context[i].copy())
            esp.threshold_edgemap(patterns[i],.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
            esp.spread_edgemap(patterns[i],edge_feature_row_breaks,edge_orientations,spread_length=5)
        all_patterns.extend(patterns)
        


template_height, template_length, \
    registered_templates,mean_template \
    = et.simple_estimate_template(all_patterns,template_length=33)

template_shape = np.array([template_height,template_length])
np.save('mean_template043012',mean_template)
np.save('template_shape043012',template_shape)


#
# get the j0 mask
#

T_mask = et.get_template_subsample_mask(mean_template,.8)

#
# do the j0 calibration on each of the examples
#
#
max_j0s = []
spread_length = 3
for pattern in all_raw_patterns_context:
    num_detections = pattern.shape[1]-template_length+1
    cur_max = -np.inf
    for frame_idx in xrange(num_detections):
            E_segment = pattern[:,frame_idx:frame_idx+template_length].copy()
            esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
            esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
            cur_max = np.maximum(cur_max,np.sum(E_segment[T_mask]))
    max_j0s.append(cur_max)        
    


#
# j0 thresh = 5000
#
#
import template_speech_rec.parts_model as pm
#exp_dir = '/home/mark/projects/Template-Speech-Recognition/Experiments/041712/'
exp_dir = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/041712/'
mean_template = np.load(exp_dir+'mean_template041712.npy')
mean_background = np.load(exp_dir+'mean_background041712.npy')
template_shape = np.load(exp_dir + 'template_shape041712.npy')
template_height = template_shape[0]
template_length = template_shape[1]



sample_rate = 16000
num_window_samples = 320
fft_length = 512
num_window_step_samples = 80
freq_cutoff = 3000
exp_path_files_dir = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/Experiments/042712/'
s_files_path_file = exp_path_files_dir+'test_path_files_s.txt'
phns_files_path_file = exp_path_files_dir+'test_path_files_phns.txt'
phn_times_files_path_file = exp_path_files_dir+'test_path_files_phn_times.txt'

# /home/mark/projects
#data_dir = "/home/mark/projects/Template-Speech-Recognition/Data/WavFilesTrain"
#data_dir = "/"
data_dir = "/var/tmp/stoehr/Projects/edges/WavFilesTrain/"
pattern = np.array(('aa','r'))
kernel_length = 7


exp = t_exp.Experiment(sample_rate,freq_cutoff,
                       num_window_samples,
                       num_window_step_samples,
                       fft_length,s_files_path_file,
                       phns_files_path_file,
                       phn_times_files_path_file,
                       data_dir,pattern,kernel_length)


parts_model = pm.PartsTriple(mean_template,.4,.4,.4)
T_mask = mean_template > .75



true_pos_thresholds = []
false_pos = []
num_frames = 0
maxima_radius = 2

spread_length = 3
all_scores = np.empty(exp.num_data,dtype = object)
mean_bgd = mean_background

bg_len = 26
abst_threshold = np.array([0.025,0.025,0.015,0.015,
                           0.02,0.02,0.02,0.02])


j0_threshold = 5500


deformed_templates = np.empty((parts_model.front_def_radius*2,parts_model.back_def_radius*2,template_height,parts_model.deformed_max_length))

for fd in xrange(-parts_model.front_def_radius,parts_model.front_def_radius):
    for bd in xrange(-parts_model.back_def_radius,parts_model.back_def_radius):
        deformed_templates[parts_model.front_def_radius+fd,
                           parts_model.back_def_radius+bd] =\
                           parts_model.get_deformed_template(fd,bd,mean_background)
        print fd,bd,deformed_templates[parts_model.front_def_radius+fd,
                           parts_model.back_def_radius+bd]
        assert (np.min(deformed_templates[parts_model.front_def_radius+fd,
                           parts_model.back_def_radius+bd]) > .000001)
                           
           

for path_idx in xrange(exp.num_data):
    print "on path", path_idx
    phns = exp.get_phns(path_idx)
    #if not exp.has_pattern(phns):
    #    continue
    phn_times = exp.get_phn_times(path_idx)
    s = exp.get_s(path_idx)
    E,edge_feature_row_breaks,\
        edge_orientations= exp.get_edgemap_no_threshold(s)
    feature_start, \
        feature_step, num_features =\
        esp._get_feature_label_times(s,
                                         exp.num_window_samples,
                                         exp.num_window_step_samples)
    feature_labels, \
        feature_label_transitions \
        = esp._get_labels(phn_times,
                      phns,
                      feature_start, feature_step, 
                      num_features,
                      exp.sample_rate)
        # add the number of frames, this helps us compute the
        # false positive rate for a given unit of time
    num_frames = num_frames + E.shape[1]
    phn_times = exp.get_phn_times(path_idx)
        # get the backgrounds for all detection spots
    num_detections = E.shape[1] - parts_model.deformed_max_length
    j0_scores = np.empty(num_detections)
    j0_detections = np.zeros(num_detections)
    j0_maxima = []
    E_bgds = []
    for frame_idx in xrange(num_detections):
        E_segment = E[:,frame_idx:frame_idx+parts_model.deformed_max_length].copy()
        esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
        j0_scores[frame_idx] = np.sum(E_segment[:,:template_length][T_mask])
        E_bgds.append(np.maximum(np.minimum(np.mean(E_segment,axis=1),
                                            .95),
                                 .05))
        if frame_idx >= template_length:
            cur_bgd = E_bgds.pop(0)
        else:
            cur_bgd = mean_background
        if frame_idx >= 2:
            if j0_scores[frame_idx-1] >= np.maximum(j0_threshold,
                                               np.maximum(j0_scores[frame_idx-2],
                                                          j0_scores[frame_idx])):
                j0_maxima.append((frame_idx,E_segment,cur_bgd))
                j0_detections[frame_idx-1] = 1
    j_detections = -np.inf * np.ones(num_detections)
    for frame_idx,E_segment,cur_bgd in j0_maxima:
        cur_max_score = -np.inf
        for fd in xrange(-parts_model.front_def_radius,parts_model.front_def_radius):
            for bd in xrange(-parts_model.back_def_radius,parts_model.back_def_radius):
                dt = deformed_templates[parts_model.front_def_radius+fd,
                           parts_model.back_def_radius+bd].copy()
                P,C =  tt.score_template_background_section(dt,cur_bgd,E_segment)
                cur_max_score = np.maximum(cur_max_score,P+C)
        j_detections[frame_idx] = cur_max_score
    print "Computed Scores"
    detection_list = [ (j_detections[d],d) for d in xrange(j_detections.shape[0])]
    detection_list = sorted(detection_list)
    detect_bool = np.empty(num_detections,dtype=bool)
    detect_bool[:] = False
    detect_bool[j_detections > -np.inf] = True
    # removing the overlapping detections
    for val,loc in detection_list:
        if detect_bool[loc]:
            detect_bool[loc+1:loc+parts_model.deformed_max_length] = False
    pattern_times = exp.get_pattern_times(phns,phn_times,s)
    maxima_idx = np.arange(num_detections)[j_detections>-np.inf]
    # see if we pick up the patterns 
    for i in xrange(len(pattern_times)):
        pattern_array =np.empty(num_detections,dtype=bool)
        pattern_array[:]=False
        # consider something a detection if its within a third of the template length around the start of the pattern
        pattern_array[pattern_times[i][0]-int(np.ceil(template_length/3)):pattern_times[i][0]+int(np.ceil(2*template_length/5.))] = True
        pattern_maxima = np.logical_and(detect_bool,pattern_array)
        if pattern_maxima.any():
            max_true_threshold = np.max(j_detections[pattern_maxima])
            true_pos_thresholds.append( 
                (max_true_threshold,path_idx))
        else:
            # this was a false negative
            true_pos_thresholds.append((-np.inf,path_idx))
            # remove the maxima that are contained within the pattern radius
            # at end of for loop only maxima left will be related to false positives
        detect_bool[pattern_times[i][0]-template_length/3:pattern_times[i][0]+template_length/3] = False
    new_false_pos = j_detections[detect_bool].shape[0]
    # check if there were false positives
    if new_false_pos:
        # will do a clustering here
        # clustered_false_pos = ps.cluster_false_positives(new_false_pos)
        # would want to  do an incremental sorting
        # otherwise we just save them to the array
        # have a tuple that gives the scores their locations and the path id
        false_pos.append((j_detections[detect_bool],
                          path_idx))



def get_threshold_array(thresholds):
    thresh = np.zeros(len(thresholds))
    for idx in xrange(thresh.shape[0]):
        thresh[idx] = thresholds[idx][0]
    thresh.sort()
    return thresh[::-1]

def get_fp_threshold_array(thresholds):
    num_fp = 0
    for idx in xrange(len(thresholds)):
        num_fp += thresholds[idx][0].shape[0]
    thresh = np.zeros(num_fp)
    for idx in xrange(len(thresholds)):
        for jdx in xrange(thresholds[idx][0].shape[0]):
            thresh[idx+jdx] = thresholds[idx][0][jdx]
    thresh.sort()
    return thresh[::-1]


# get simple threshold arrays
num_examples = len(true_pos_thresholds)
pthresh = get_threshold_array(true_pos_thresholds)
fpthresh = get_fp_threshold_array(false_pos)


# start writing the 
fp_rates = np.zeros(num_examples+1,dtype=float)
fp_idx = 0
cur_num_fp = 0
p_rates = np.zeros(num_examples,dtype=float)
# handle the fact that there might be some thresholds that are above any of the false positives
pthresh_start_idx = 0
while fpthresh[fp_idx] < pthresh[pthresh_start_idx]:
    pthresh_start_idx +=1

fp_rates[:pthresh_start_idx] = 0.
for ex_id in xrange(pthresh_start_idx,num_examples):
    #invariant should be that
    #fpthresh[fp_idx] >= pthresh[ex_id]
    # as we increment ex_id we move fp_idx forward by sufficiently many to preserve that relationship
    while fpthresh[fp_idx] >= pthresh[ex_id]:
        fp_idx+=1
    # now fpthresh[fp_idx] < pthresh[ex_id]
    fp_rates[ex_id] = fp_idx/float(num_frames)

for id in xrange(ex_id,num_examples):
    fp_rates[id] = fp_rates[ex_id-1]

# the experiment went quite poorly with very low detection rates,
# going to do something further which will see how the parts fitting can work on the positive examples



import pylab as pl

pl.figure()
pl.plot(np.arange(num_examples)/float(num_examples),all_fp_rates[0]/.005,'ro')
pl.ylabel('false positives per second')
pl.xlabel('true positive rate')
pl.legend()
pl.title = ("ROC curve for parts model")
pl.show()


