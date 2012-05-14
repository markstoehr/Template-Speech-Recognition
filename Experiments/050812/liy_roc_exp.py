root_path = '/home/mark/projects/Template-Speech-Recognition/'

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
            assert len(base_object.shape) == 2
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
        elif tpm.__class__ == TwoPartModel:
            template_height, template_length = tpm.bg.shape[0], tpm.length_range[1]
            if bg:
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
                max([ self._score_sub_no_bg(E_window,t_id) for t_id in xrange(self.template.num_templates)])
            # just create a uniform background with .4 as the edge frequency
            self.coarse_template = get_coarse_segment(self.template.def_templates[-self.template.min_max_def[0]],
                                                      coarse_type="avg",
                                                      coarse_factor = self.coarse_factor)
            self.coarse_template_mask = self.coarse_template > .7
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

            

def get_roc_coarse(dat_iter,pattern, classifier,
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
    """
    for datum_id in xrange(tune_data_iter.num_data):
        if datum_id % 10 == 0:
            print "working on example", datum_id
        if tune_data_iter.next(compute_pattern_times=True,
                            max_template_length=classifier.window[1]):
            pattern_times = tune_data_iter.pattern_times
            num_detections = tune_data_iter.E.shape[1] - classifier.window[1]
            detect_scores = -np.inf * np.ones(num_detections)
            for d in xrange(num_detections):
                E_segment = tune_data_iter.E[:,d:d+classifier.window[1]].copy()                
                classifier.coarse_score_count
                


            
