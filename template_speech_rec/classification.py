import sys, os

#import template_speech_rec.data_loading as dl
import numpy as np
import template_experiments as template_exp
import edge_signal_proc as esp
import estimate_template as et
import test_template as tt
import parts_model as pm

class Classifier:
    def __init__(self,base_object, coarse_factor=2,
                 coarse_template_threshold = .5,
                 bg = None):
        self.coarse_factor = coarse_factor
        if base_object.__class__ is list:
            # we assume that all objects have the same
            # length so we are comparing apples to apples
            template_height = base_object[0].shape[0]
            template_length = max([t.shape[1] for t in base_object])
            self.type = "collection"
            self.window = np.array((template_height,template_length))
            self.template = base_object
            self.bg = bg
            self.pad_template = lambda tmplt, bg: np.hstack((tmplt,
                                                             np.tile(
                        bg.reshape(self.window[0],1),
                        (1,self.window[1]-tmplt.shape[1]))))
            self.score = lambda E_window,bg:\
                max([sum(tt.score_template_background_section(
                            self.pad_template(tmplt,bg),
                            bg,E_window)) for tmplt in self.template])
            self.score_no_bg = lambda E_window:\
                max([sum(tt.score_template_background_section(
                            self.pad_template(tmplt,self.bg),
                            self.bg,
                            E_window)) for tmplt in self.template])
            self.coarse_template = [get_coarse_segment(
                    tmplt,
                    coarse_type="avg",
                    coarse_factor = self.coarse_factor) for tmplt in self.template]
            self.coarse_length = self.coarse_template[0].shape[1]
            self.coarse_template_mask = [T >.7 for T in self.coarse_template]
            self.coarse_score_like = lambda E_window,bg:\
                max([sum(tt.score_template_background_section(T,
                                                              bg,
                            get_coarse_segment(E_window,
                                     coarse_type='max',
                                     coarse_factor=self.coarse_factor))) for T in self.coarse_template])
            self.coarse_score_like_no_bg = lambda E_window:\
                max([sum(tt.score_template_background_section(T,
                            self.bg,
                            get_coarse_segment(E_window,
                                     coarse_type='max',
                                     coarse_factor=self.coarse_factor))) for T in self.coarse_template])
            self.coarse_score_count = lambda E_window:\
                max([np.sum(get_coarse_segment(E_window,
                               coarse_type='max',
                               coarse_factor=self.coarse_factor)[T_mask]) for T_mask in self.coarse_template_mask])
        if base_object.__class__ == np.ndarray:
            # this means that we just have a template which will be a
            # 2-d ndarray, our function assumes this
            template_height, template_length = base_object.shape
            self.type = "template"
            self.window = np.array((template_height,template_length))
            self.template = base_object
            self.bg = bg
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
        elif base_object.__class__ == pm.TwoPartModel:
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
        
        def score_pad(E_window,bg):
            diff = E_window.shape[1] - self.window[1]
            if diff >= 0:
                return max(map(lambda loc:, 
                               self.score(
                            E_window[:,
                                loc:loc+self.window[1]],
                            bg),
                               np.arange(diff,dtype=int)))
            else: # diff < 0
                # padd E_window with background
                return self.score( np.hstack((E_window,
                                       np.tile(bg,(-diff,1)).T)))



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


def get_roc_generous(data_iter, classifier,coarse_thresh=-np.inf,
                   allowed_overlap = .1,
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
    Computes an ROC curve for a classifier, do not remove positive examples that overlap with negative examples, simply take the max within the positive regions
    """
    num_frames = 0
    all_positive_scores = []
    all_negative_scores = []
    all_positive_counts = []
    all_negative_counts = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            coarse_count_scores = -np.inf *np.ones(num_detections)
            coarse_scores = -np.inf * np.ones(num_detections)
            esp._edge_map_threshold_segments(data_iter.E,
                                 classifier.window[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = data_iter.edge_orientations,
                                 edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]]
                scores[d] = classifier.score_no_bg(E_segment)
                coarse_count_scores[d] = classifier.coarse_score_count(E_segment)
                if d>1 and d<num_detections-1:
                    if (coarse_count_scores[d-1] > coarse_thresh) and \
                            ((coarse_count_scores[d-1]>\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d-2]) or\
                                (coarse_count_scores[d-1]>=\
                                  coarse_count_scores[d] and\
                                  coarse_count_scores[d-1]>\
                                  coarse_count_scores[d-2]) ):
                        coarse_scores[d] = classifier.score_no_bg(E_segment)
            # get the positive and negative scores removed out
            pos_counts =[]
            pos_scores = []
            neg_indices = np.empty(scores.shape[0],dtype=bool)
            neg_indices[:]=True
            for pt in xrange(len(pattern_times)):
                pos_scores.append(np.max(scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                pos_counts.append(np.max(coarse_scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                neg_indices[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][1]] = False
            # get rid of overlapping instances
            neg_indices_counts_non_overlap = remove_overlapping_examples(np.argsort(coarse_scores),
                                                        classifier.coarse_length,
                                                        int(allowed_overlap*classifier.coarse_length))
            neg_indices_non_overlap = remove_overlapping_examples(np.argsort(scores),
                                                        classifier.window[1],
                                                        int(allowed_overlap*classifier.window[1]))

            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_non_overlap] =True
            neg_indices_full = np.logical_and(neg_indices,neg_idx2)
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_counts_non_overlap] =True
            neg_indices_coarse = np.logical_and(neg_indices,neg_idx2)
            all_positive_scores.extend(pos_scores)
            all_positive_counts.extend(pos_counts)
            all_negative_scores.extend(scores[neg_indices_full])
            all_negative_counts.extend(coarse_scores[neg_indices_coarse])
            
        else:
            break
    like_roc = get_roc(np.sort(all_positive_scores)[::-1],
                        np.sort(all_negative_scores)[::-1],
                        num_frames)
    count_roc = get_roc(np.sort(all_positive_counts)[::-1],
                        np.sort(all_negative_counts)[::-1],
                        num_frames)
    return like_roc, count_roc


            
def get_pos_neg_scores(inds,pattern_times,scores,window_length):
    """
    Ouputs the scores for the positive examples and the negative
    examples.  The positive examples, by default, are assigned a score
    of negative infinity. The number such examples is given by pattern
    times.
    """
    pos_scores, neg_scores = [],[]
    pos_patterns = np.empty(scores.shape[0],dtype=int)
    pos_patterns[:] = 0
    pos_scores = [-np.inf] * len(pattern_times)
    for pt in xrange(len(pattern_times)):
        pos_patterns[pattern_times[pt][0]-int(np.ceil(window_length/3.)):
                         pattern_times[pt][0]+int(np.ceil(window_length/3.))]= pt+1
    for ind in inds:
        if pos_patterns[ind]>0:
            if scores[ind] > pos_scores[pos_patterns[ind]-1]:
                pos_scores[pos_patterns[ind]-1] = scores[ind]
        else:
            neg_scores.append(scores[ind])
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
                if cur_neg_idx >= neg.shape[0]:
                    end_loop = True
                    break
        if end_loop:
            for roc_idx_prime in xrange(roc_idx,roc_vals.shape[0]):
                roc_vals[roc_idx_prime] = -np.inf
            break
        else:
            roc_vals[roc_idx] = cur_neg_idx/num_frames        
    return lambda false_neg_rate: (roc_vals[int(false_neg_rate*pos.shape[0])],pos[int(false_neg_rate*pos.shape[0])]), roc_vals

def get_classification_roc(data_iter,classifier):
    """
    Returns an roc for the given classification task
    """
    num_frames=0
    all_positive_scores = []
    all_negative_scores = []
    data_iter.reset_exp()
    for datum_id in xrange(data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1],
                          wait_for_positive_example=True):
            pattern_times = data_iter.pattern_times
            num_detections = data_iter.E.shape[1] - classifier.window[1]
            num_frames += data_iter.E.shape[1]
            scores = -np.inf * np.ones(num_detections)
            esp._edge_map_threshold_segments(data_iter.E,
                                 classifier.window[1],
                                 1, 
                                 threshold=.3,
                                 edge_orientations = data_iter.edge_orientations,
                                 edge_feature_row_breaks = data_iter.edge_feature_row_breaks)
            for d in xrange(num_detections):
                E_segment = data_iter.E[:,d:d+classifier.window[1]]
                scores[d] = classifier.score_no_bg(E_segment)
            # get the positive and negative scores removed out
            pos_scores = []
            neg_indices = np.empty(scores.shape[0],dtype=bool)
            neg_indices[:]=True
            for pt in xrange(len(pattern_times)):
                pos_scores.append(np.max(scores[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][0]+int(np.ceil(classifier.window[1]/3.))]))
                neg_indices[pattern_times[pt][0]-int(np.ceil(classifier.window[1]/3.)):pattern_times[pt][1]] = False
            # get rid of overlapping instances
            neg_indices_non_overlap = remove_overlapping_examples(np.argsort(scores),
                                                        classifier.window[1],
                                                        int(allowed_overlap*classifier.window[1]))
            neg_idx2 = np.empty(scores.shape[0],dtype=bool)
            neg_idx2[neg_indices_non_overlap] =True
            neg_indices_full = np.logical_and(neg_indices,neg_idx2)
            all_positive_scores.extend(pos_scores)
            all_negative_scores.extend(scores[neg_indices_full])
        else:
            break
    return get_roc(np.sort(all_positive_scores)[::-1],
                        np.sort(all_negative_scores)[::-1],
                        num_frames)
    
