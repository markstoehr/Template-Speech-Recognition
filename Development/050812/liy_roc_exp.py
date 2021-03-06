root_path = '/var/tmp/stoehr/Projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)

#import template_speech_rec.data_loading as dl
import numpy as np
import template_speech_rec.template_experiments as template_exp
import template_speech_rec.edge_signal_proc as esp
import template_speech_rec.estimate_template as et
import template_speech_rec.test_template as tt



edge_feature_row_breaks = np.load('edge_feature_row_breaks.npy')

edge_orientations = np.load('edge_orientations.npy')
abst_threshold = np.load('abst_threshold.npy')


class Classifier:
    def __init__(self,base_object, coarse_factor=2,
                 coarse_template_threshold = .7,
                 bg = None):
        self.coarse_factor = coarse_factor
        if base_object.__class__ == np.ndarray:
            # this means that we just have a template which will be a
            # 2-d ndarray, our function assumes this
            template_height, template_length = base_object.shape
            self.type = "template"
            self.window = np.array((template_height,template_length))
            self.template = base_object
            self.score = lambda E_window,bg:\
                sum(tt.score_template_background_section(self.template,
                                                         bg,E_window))
            # just create a uniform background with .4 as the edge frequency
            self.score_no_bg = lambda E_window:\
                sum(tt.score_template_background_section(self.template,
                                                         self.bg,
                                                         E_window))
            self.coarse_template = get_coarse_segment(self.template,
                                                      coarse_type="avg",
                                                      coarse_factor = self.coarse_factor)
            self.coarse_length = self.coarse_template.shape[1]
            self.coarse_template_mask = self.coarse_template > .7
            self.coarse_score_like = lambda E_window,bg:\
                sum(tt.score_template_background_section(self.coarse_template,
                            bg,
                            get_coarse_segment(E_window,
                                     coarse_type='max',
                                     coarse_factor=self.coarse_factor)))
            self.coarse_score_like_no_bg = lambda E_window:\
                sum(tt.score_template_background_section(self.coarse_template,
                            self.bg,
                            get_coarse_segment(E_window,
                                     coarse_type='max',
                                     coarse_factor=self.coarse_factor)))
            self.coarse_score_count = lambda E_window:\
                np.sum(get_coarse_segment(E_window,
                               coarse_type='max',
                               coarse_factor=self.coarse_factor)[self.coarse_template_mask])
        elif base_object.__class__ == TwoPartModel:
            template_height, template_length = base_object.bg.shape[0],base_object.length_range[1]
            if bg is not None:
                self.bg = bg
            else:
                self.bg = .4 * np.ones(template_height)
            self.type = "TwoPartModel"
            self.window = np.array((template_height,template_length))
            self.template = base_object
            self._score_sub_no_bg = lambda E_window,t_id:\
                sum(tt.score_template_background_section(self.template.def_templates[t_id],
                                                         self.template.bg,E_window))
            self.score_no_bg = lambda E_window:\
                max([ self._score_sub_no_bg(E_window,t_id) for t_id in xrange(self.template.def_templates.shape[0])])
            # just create a uniform background with .4 as the edge frequency
            self.coarse_template = get_coarse_segment(self.template.base_template,
                                                      coarse_type="avg",
                                                      coarse_factor = self.coarse_factor)
            self.coarse_length = self.coarse_template.shape[1]
            self.coarse_template_mask = self.coarse_template > .7
            self.coarse_score_like_no_bg = lambda E_window:\
                sum(tt.score_template_background_section(self.coarse_template,
                            self.bg,
                            get_coarse_segment(E_window,
                                     coarse_type='max',
                                     coarse_factor=self.coarse_factor)[:,:self.coarse_length]))
            self.coarse_score_count = lambda E_window:\
                np.sum(get_coarse_segment(E_window,
                               coarse_type='max',
                               coarse_factor=self.coarse_factor)[:,:self.coarse_length][self.coarse_template_mask])

        




# function for getting a coarser template to do a classifier
# cascade
def get_coarse_segment(segment, coarse_type="max",
                        coarse_factor=2):
    """
    Create a coarse segment/widnow, the coarser template/window is
    shorter in time, in general, multiple frames of the original
    template/detection window are mapped to a single frame in the
    coarse template/detection window. Templates have probabilities as
    their entries and for the coarse template we take their average
    when multiple frames are mapped to a single frame. Coarser windows
    will generally use a max operation to pick whether there is an
    edge feature present at that point
    """
    height,length = segment.shape
    coarse_length = int(length/coarse_factor)
    coarse_segment = np.zeros((height,coarse_length))
    for frame_id in xrange(coarse_length):
        lo,hi = map_coarse_frame_to_range(frame_id,
                                          coarse_factor,
                                          length)
        if coarse_type == "avg":
            coarse_segment[:,frame_id] = \
                np.mean(segment[:,lo:hi],axis=1)
        elif coarse_type == "max":
            coarse_segment[:,frame_id] = \
                np.max(segment[:,lo:hi],axis=1)
    return coarse_segment

def map_coarse_frame_to_range(frame_id,
                              coarse_factor,
                              length):
    """
    When we create a coarser template, in general, multiple frames in
    the template get mapped to the same frame in the coarser template
    this function tells us for a given coarse template frame what
    frames in the base template get mapped
    
    Parameters:
    ===========
    frame_id: 
        The frame we're working with
    coarse_factor:
        The coarsening factor, gives us a sense of how big 
        the range should be
    length:
        This is the length fo the template, we don't want to
        exceed this length.

    Output:
    =======
    lo:
        the index prior to the first frame in the base 
        template that gets mapped to frame_id.
    hi:
        the index following the last frame in the base
        template that gets mapped to frame_id

    Note:
    We are assuming here that we are thinking in the
    pythonic list model where list indices refer to the
    points in between list entries (and at the front and
    back), so there are n+1 positions to refer to if the
    list (or array) has length n.
        
    """
    return (coarse_factor * frame_id,  # lo
            min(coarse_factor  * (frame_id+1), # hi
                length))



#
# Testing the coarse aspects of the classifier here
#    get a template and a two-part model
#




# function for getting a coarse window        

            

def get_roc_coarse(data_iter, classifier,
                   allowed_overlap = .2,
            edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.]),
            edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                        [-1.,  1.]]),
            abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02]),
            spread_radius=3):
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positive_counts = []
    all_positive_likes = []
    all_negative_counts = []
    all_negative_likes = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            coarse_count_scores = -np.inf * np.ones(num_detections)
            coarse_like_scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                coarse_count_scores[d] = classifier.coarse_score_count(E_segment)
                coarse_like_scores[d] = classifier.coarse_score_like_no_bg(E_segment)
            # now we get the indices sorted
            count_indices = remove_overlapping_examples(np.argsort(coarse_count_scores),
                                                        classifier.coarse_length,
                                                        int(allowed_overlap*classifier.coarse_length))
            like_indices = remove_overlapping_examples(np.argsort(coarse_like_scores),
                                                       classifier.coarse_length,
                                                       int(allowed_overlap*classifier.coarse_length))
            positive_counts, negative_counts =  get_pos_neg_scores(count_indices,pattern_times,
                                                                     coarse_count_scores)
            positive_likes, negative_likes = get_pos_neg_scores(like_indices,pattern_times,
                                                                coarse_like_scores)
            all_positive_counts.extend(positive_counts)
            all_negative_counts.extend(negative_counts)
            all_positive_likes.extend(positive_likes)
            all_negative_likes.extend(negative_likes)
    count_roc = get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
    like_roc = get_roc(np.sort(all_positive_likes)[::-1], 
                       np.sort(all_negative_likes)[::-1],
                       num_frames)
    return count_roc, like_roc
    

def get_roc_full(data_iter, classifier, coarse_thresh,
                   allowed_overlap = .2,
            edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.]),
            edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                        [-1.,  1.]]),
            abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02]),
            spread_radius=3):
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positive_counts = []
    all_positive_likes = []
    all_negative_counts = []
    all_negative_likes = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                          wait_for_true
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                coarse_count_score = classifier.coarse_score_count(E_segment)
                if coarse_count_score > coarse_thresh:
                    scores[d] = classifier.score_no_bg(E_segment)
            # now we get the indices sorted
            score_indices = remove_overlapping_examples(np.argsort(scores),
                                                        classifier.window[1],
                                                        int(allowed_overlap*classifier.coarse_length))
            positive_scores, negative_scores =  get_pos_neg_scores(score_indices,pattern_times,
                                                                     scores)
            positive_likes, negative_likes = get_pos_neg_scores(like_indices,pattern_times,
                                                                coarse_like_scores)
            all_positive_counts.extend(positive_counts)
            all_negative_counts.extend(negative_counts)
            all_positive_likes.extend(positive_likes)
            all_negative_likes.extend(negative_likes)
        else:
            break
    count_roc = get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
    like_roc = get_roc(np.sort(all_positive_likes)[::-1], 
                       np.sort(all_negative_likes)[::-1],
                       num_frames)
    return count_roc, like_roc

            
def get_pos_neg_scores(inds,pattern_times,coarse_scores,window_length):
    """
    Ouputs the scores for the positive examples and the negative
    examples.  The positive examples, by default, are assigned a score
    of negative infinity. The number such examples is given by pattern
    times.
    """
    pos_scores, neg_scores = [],[]
    pos_patterns = np.empty(coarse_scores.shape[0],dtype=int)
    pos_patterns[:] = 0
    pos_scores = [-np.inf] * len(pattern_times)
    for pt in xrange(len(pattern_times)):
        pos_patterns[pattern_times[pt][0]-window_length/3:pattern_times[pt][0]+window_length/3]= pt
    for ind in inds:
        if pos_patterns[ind]>0:
            if coarse_scores[ind] > pos_scores[pos_patterns[ind]]:
                pos_scores[pos_patterns[ind]] = coarse_scores[ind]
        else:
            neg_scores.append(coarse_scores[ind])
    return pos_scores,neg_scores
    


def remove_overlapping_examples(score_inds,length,overlap):
    available_times = np.empty(score_inds.shape[0],dtype=bool)
    available_times[:] = True
    detections = []
    for ind in score_inds:
        if available_times[ind]:
            detections.append(ind)
            available_times[ind+overlap-length+1:ind+length-overlap] = False
    return detections
    
            
def get_roc(pos,neg,num_frames):
    """
    pos and neg are numpy arrays and assumed to be sorted
    in descending order
    """
    num_frames = float(num_frames)
    roc_vals = np.zeros(len(pos))
    cur_neg_idx = 0
    while pos[0] <= neg[cur_neg_idx]: cur_neg_idx += 1
    roc_vals[0] = cur_neg_idx/num_frames
    end_loop = False
    for roc_idx in xrange(1,roc_vals.shape[0]):
        if pos[roc_idx] < neg[-1]:
            end_loop = True
        else:
            while pos[roc_idx] <= neg[cur_neg_idx]:
                cur_neg_idx +=1
                if cur_neg_idx < neg.shape[0]:
                    end_loop = True
                    break
        if end_loop:
            for roc_idx_prime in xrange(roc_idx,roc_vals.shape[0]):
                roc_vals[roc_idx_prime] = -np.inf
            break
        else:
            roc_vals[roc_idx] = cur_neg_idx/num_frames        
    return lambda false_neg_rate: (roc_vals[int(false_neg_rate*pos.shape[0])],pos[int(false_neg_rate*pos.shape[0])]), roc_vals


texp = template_exp.\
    Experiment(pattern=np.array(('l','iy')),
               data_paths_file=root_path+'Data/WavFilesTrainPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )




train_data_iter, tune_data_iter =\
    template_exp.get_exp_iterator(texp,train_percent=.6)

liy_patterns = []
train_data_iter.reset_exp()
for datum_id in xrange(train_data_iter.num_data):
    if datum_id % 10 == 0:
        print datum_id
    if train_data_iter.next(wait_for_positive_example=True,
                            compute_patterns=True,
                            max_template_length=40):
        # the context length is 11
        for p in train_data_iter.patterns:
            pattern = p.copy()
            esp.threshold_edgemap(pattern,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
            esp.spread_edgemap(pattern,edge_feature_row_breaks,edge_orientations,spread_length=3)
            liy_patterns.append(pattern)
    else:
        break

_,_ ,\
    registered_ex_liy,liy_template \
    = et.simple_estimate_template(liy_patterns)

np.save('registered_ex_liy051512',registered_ex_liy)
np.save('liy_template051512',liy_template)



mean_background = np.load('mean_background_liy051012.npy')

template_shape = liy_template.shape
tpm_liy = TwoPartModel(liy_template,mean_background,
                       2*template_shape[1]/3,)




classifier = Classifier(tpm_liy,bg=mean_background)

count_roc, like_roc = get_roc_coarse(tune_data_iter, tpm_Classifier)

data_iter = tune_data_iter
if True:
    allowed_overlap = .2
    edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.])
    edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positive_counts = []
    all_positive_likes = []
    all_negative_counts = []
    all_negative_likes = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(wait_for_positive_example=True,
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            coarse_count_scores = -np.inf * np.ones(num_detections)
            coarse_like_scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                coarse_count_scores[d] = classifier.coarse_score_count(E_segment)
                coarse_like_scores[d] = classifier.coarse_score_like_no_bg(E_segment)
            # now we get the indices sorted
            count_indices = remove_overlapping_examples(np.argsort(coarse_count_scores),
                                                        classifier.coarse_length,
                                                        int(allowed_overlap*classifier.coarse_length))
            like_indices = remove_overlapping_examples(np.argsort(coarse_like_scores),
                                                       classifier.coarse_length,
                                                       int(allowed_overlap*classifier.coarse_length))
            positive_counts, negative_counts =  get_pos_neg_scores(count_indices,pattern_times,
                                                                     coarse_count_scores)
            positive_likes, negative_likes = get_pos_neg_scores(like_indices,pattern_times,
                                                                coarse_like_scores)
            all_positive_counts.extend(positive_counts)
            all_negative_counts.extend(negative_counts)
            all_positive_likes.extend(positive_likes)
            all_negative_likes.extend(negative_likes)
        else:
            break
    count_roc = get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
    like_roc = get_roc(np.sort(all_positive_likes)[::-1], 
                       np.sort(all_negative_likes)[::-1],
                       num_frames)
    return count_roc, like_roc


texp_test = template_exp.\
    Experiment(pattern=np.array(('l','iy')),
               data_paths_file=root_path+'Data/WavFilesTestPaths_feverfew',
               spread_length=3,
               abst_threshold=.0001*np.ones(8),
               fft_length=512,num_window_step_samples=80,
               freq_cutoff=3000,sample_rate=16000,
               num_window_samples=320,kernel_length=7
               )


_, coarse_thresh = count_roc(.97)


test_data_iter, _ =\
    template_exp.get_exp_iterator(texp_test,train_percent=1.)


data_iter = test_data_iter
if True:
    allowed_overlap = .2
    edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.])
    edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positives = []
    all_negatives = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(wait_for_positive_example=True,
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                coarse_count_score = classifier.coarse_score_count(E_segment)
                if coarse_count_score > coarse_thresh:
                    scores[d] = classifier.score_no_bg(E_segment)
            # now we get the indices sorted
            indices = remove_overlapping_examples(np.argsort(scores),
                                                  classifier.window[1],
                                                  int(allowed_overlap*classifier.window[1]))
            positives, negatives =  get_pos_neg_scores(indices,pattern_times,
                                                                     scores)
            all_positives.extend(positives)
            all_negatives.extend(negatives)
        else:
            break
    liy_roc,liy_roc_vals = get_roc(np.sort(all_positives)[::-1],
                        np.sort(all_negatives)[::-1],
                        num_frames)


#

#
#

data_iter = tune_data_iter
if True:
    allowed_overlap = .2
    edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.])
    edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positives = []
    all_negatives = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(wait_for_positive_example=True,
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                coarse_count_score = classifier.coarse_score_count(E_segment)
                if coarse_count_score > coarse_thresh:
                    scores[d] = classifier.score_no_bg(E_segment)
            # now we get the indices sorted
            indices = remove_overlapping_examples(np.argsort(scores),
                                                  classifier.window[1],
                                                  int(allowed_overlap*classifier.window[1]))
            positives, negatives =  get_pos_neg_scores(indices,pattern_times,
                                                                     scores)
            all_positives.extend(positives)
            all_negatives.extend(negatives)
        else:
            break
    liy_roc,liy_roc_vals = get_roc(np.sort(all_positives)[::-1],
                        np.sort(all_negatives)[::-1],
                        num_frames)



#setting the threshold again
# seems that things got messed up

classifier = Classifier(tpm_liy,bg=mean_background,coarse_factor=1)


data_iter = test_data_iter
if True:
    allowed_overlap = .2
    edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.])
    edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positives = []
    all_negatives = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(wait_for_positive_example=True,
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                scores[d] = classifier.score_no_bg(E_segment)
            # now we get the indices sorted
            indices = remove_overlapping_examples(np.argsort(scores),
                                                  classifier.window[1],
                                                  int(allowed_overlap*classifier.window[1]))
            positives, negatives =  get_pos_neg_scores(indices,pattern_times,
                                                                     scores,classifier.window[1])
            all_positives.extend(positives)
            all_negatives.extend(negatives)
        else:
            break
    liy_roc_full,liy_roc_vals_full = get_roc(np.sort(all_positives)[::-1],
                                                 np.sort(all_negatives)[::-1],
                                                 num_frames)




#
#
# getting such terrible results we want to make sure that everything being done is kosher, namely we need to make sure that the template being used actually gets reasonable results


#


data_iter = tune_data_iter
if True:
    allowed_overlap = .8
    edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.])
    edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_positives = []
    all_negatives = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(wait_for_positive_example=True,
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - liy_template.shape[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+liy_template.shape[1]].copy()                
                esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                scores[d] = sum(tt.score_template_background_section(liy_template,
                                                         mean_background,E_segment))
            # now we get the indices sorted
            indices = remove_overlapping_examples(np.argsort(scores),
                                                  liy_template.shape[1],
                                                  int(allowed_overlap*liy_template.shape[1]))
            positives, negatives =  get_pos_neg_scores(indices,pattern_times,
                                                                     scores,classifier.window[1])
            all_positives.extend(positives)
            all_negatives.extend(negatives)
        else:
            break
    liy_roc_full,liy_roc_vals_full = get_roc(np.sort(all_positives)[::-1],
                                                 np.sort(all_negatives)[::-1],
                                                 num_frames)



#
#
# going to try to see where maxima detections are relative to the location of the true positives
#
#

data_iter = tune_data_iter
if True:
    allowed_overlap = .8
    edge_feature_row_breaks= np.array([   0.,   
                                               45.,   
                                               90.,  
                                               138.,  
                                               186.,  
                                               231.,  
                                               276.,  
                                               321.,  
                                               366.])
    edge_orientations=np.array([[ 1.,  0.],
                                        [-1.,  0.],
                                        [ 0.,  1.],
                                        [ 0., -1.],
                                        [ 1.,  1.],
                                        [-1., -1.],
                                        [ 1., -1.],
                                [-1.,  1.]])
    abst_threshold=np.array([.025,.025,.015,.015,
                                      .02,.02,.02,.02])
    spread_radius=3
    """
    Find the appropriate threshold for the coarse classifier, this
    should be run on tuning data, and then we can get a level for the
    tradeoff between false positives and false negatives the first
    pair is the roc curve for the count test and the second is for the
    coarse likelihood test

    The way this works is that we find the scores for each window and
    then we rank them and remove overlapping windows that are lower
    rank if the overlap is below a certain percentage

    We are assuming that the classifier has been implemented and initialized properly
    """
    num_frames = 0
    all_scores = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(wait_for_positive_example=True,
                          compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - liy_template.shape[1]
            num_frames += data_iter.E.shape[1]
            scores_list = \
                [-np.inf * np.ones(max(liy_template.shape[1],pattern_time[1])-(pattern_time[0]-liy_template.shape[1]))\
                      for pattern_time in pattern_times]
            for pt in xrange(len(scores_list)):
                for d in xrange(pattern_times[pt][0]-liy_template.shape[1],max(liy_template.shape[1],pattern_times[pt][1])):
                    E_segment = data_iter.E[:,d:d+liy_template.shape[1]].copy()                
                    if E_segment.shape[1] != liy_template.shape[1]:
                        scores_list[pt][d-(pattern_times[pt][0]-liy_template.shape[1])] = -np.inf
                        continue
                    esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,report_level=False,abst_threshold=abst_threshold)
                    esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=3)
                    scores_list[pt][d-(pattern_times[pt][0]-liy_template.shape[1])] = sum(tt.score_template_background_section(liy_template,
                                                         mean_background,E_segment))
            # now we get the indices sorted
            all_scores.extend(zip(scores_list,
                                  [pattern_times[p][1]-pattern_times[p][0] \
                                       for p in xrange(len(pattern_times))]))
        else:
            break



#
#
# it seems that something is entirely amiss right here and we have messed up the code some how
#
#

#
# going to try to run a regression experiment
#
# look at the top three values and then see if I can fit a regression function

top_score_length_cmp = np.empty((len(all_scores),7))
for d in xrange(top_score_length_cmp.shape[0]):
    top_score_length_cmp[d][6] = all_scores[d][1]
    sorted_idx = np.argsort(all_scores[d][0])
    for i in xrange(3):
        top_score_length_cmp[d][i] = all_scores[d][0][sorted_idx[i]]
        top_score_length_cmp[d][i+3] = sorted_idx[i] - liy_template.shape[1]

np.save('tslc',top_score_length_cmp)

