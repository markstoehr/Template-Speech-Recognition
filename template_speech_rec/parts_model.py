#
# Author: Mark Stoehr, 2012
#
# Developing a parts model
# Begin Middle End
#
#
# todo, find a better way to make arrays
#       what is the best way to make arrays in numpy from a
#       list that you don't know how long it is

import numpy as np
import test_template as tt
import edge_signal_proc as esp


class PartsTriple:
    def __init__(self,base_template,front_fraction,middle_fraction,back_fraction):
        # create an overlapping part model
        self.base_template = base_template
        self.height,length = base_template.shape
        self.front_length = int(front_fraction * length)
        self.front = base_template[:,:self.front_length].copy()
        self.front_start = 0        
        #
        # Get the back now
        # int(back_fraction *length) 
        # = back_length 
        # = length-back_start = end_idx-back_start +1
        
        self.back_length = int(back_fraction*length)
        self.back = base_template[:,-self.back_length:].copy()
        self.back_start = length - self.back_length
        #
        #  middle section
        #
        self.middle_length = int(middle_fraction*length)
        self.middle_start = length/2 - self.middle_length/2
        self.middle = base_template[:,self.middle_start:\
                                          self.middle_start+self.middle_length].copy()
        # def limits are deformation limits
        #   should be such that
        # front+min_def+front_length should be equal to middle_start
        # 0+ max_def = middle_start
        self.front_def_radius = self.front_length-self.middle_start
        # back_start+max_def = middle_start+middle_length
        self.back_def_radius = self.middle_start+self.middle_length - self.back_start
        # this determines how long the match is
        self.deformed_max_length = self.back_start + self.front_def_radius+self.back_def_radius+self.back_length-1


    def get_deformed_template(self,front_displace,back_displace,bgd):
        front_displace
        deformed_template = np.zeros((self.height,
                                      self.deformed_max_length))
        front_start = 0
        front_end = front_start + self.front_length
        middle_start = self.middle_start -front_displace
        middle_end = middle_start + self.middle_length
        back_start = self.back_start + back_displace - front_displace
        back_end = back_start + self.back_length
        assert (middle_start <= front_end)
        assert (back_start <= middle_end)
        # set the location of the background
        if back_end < self.deformed_max_length:
            deformed_template[:,back_end:] \
                = np.tile(bgd,
                          (self.deformed_max_length-back_end,1)).transpose()
        deformed_template[:,front_start:front_end] = self.front.copy()
        #print "front_min:",np.min(deformed_template[:,front_start:front_end])
        deformed_template[:,back_start:back_end] = self.back.copy()
        #print "back_min:",np.min(deformed_template[:,back_start:back_end])
        # now handle overlaps between the front and middle
        # then the overlaps between the middle and back
        front_overlap = front_end - middle_start
        if front_overlap > 0:
            deformed_template[:,front_end-front_overlap:front_end]=(
                self.front[:,-front_overlap:]
                + self.middle[:,:front_overlap])/2.
        
        back_overlap = middle_end- back_start
        if back_overlap >0:
            deformed_template[:,middle_end-back_overlap:middle_end]=(
                self.middle[:,-back_overlap:]
                + self.back[:,:back_overlap])/2.
            
        # set the undeformed middle section
        deformed_template[:,middle_start+front_overlap:middle_end-back_overlap] = self.middle[:,front_overlap:-back_overlap].copy()
        return deformed_template


    def get_hyp_segment(self,E,E_loc,
                            edge_feature_row_breaks,
                            abst_threshold,
                            edge_orientations,
                            spread_length=3):
        E_segment = E[:,E_loc:
                            E_loc+self.deformed_max_length].copy()
        esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)
        return E_segment

    def get_fit_loc(self,E_loc,front_displace):
        return E_loc +front_displace


    def get_hyp_segment_bgd(self,E,E_loc,
                            edge_feature_row_breaks,
                            abst_threshold,
                            edge_orientations,
                            spread_length=3,bg_len=26):
        E_segment = E[:,E_loc-self.front_def_radius-bg_len:
                            E_loc-self.front_def_radius]
        esp.threshold_edgemap(E_segment,.30,edge_feature_row_breaks,abst_threshold=abst_threshold)
        esp.spread_edgemap(E_segment,edge_feature_row_breaks,edge_orientations,spread_length=spread_length)

        
    
    def score_def_template(self,hyp_segment,def_template,bgd):
        P,C = tt.score_template_background_section(def_template,bgd,hyp_segment)
        return P+C
    
    def fit_template(E,E_loc,bgd):
        # will find the best fit for a given level of displacement
        
        pass


class PartsTripleContext:
    def __init__(self,template,front,back,middle_start_loc,front_start_loc,back_start_loc):
        # assert(len(template.shape) == 2)
        self.height, self.length = template.shape
        # should be zero as this is the root part
        self.middle_start = middle_start_loc
        self.front_height, self.front_length = front.shape
        # should be negative
        self.front_start = front_start_loc
        # where the front part and the template align
        self.back_height, self.back_length = back.shape
        # where the back part and the template align
        self.back_start = back_start_loc
        self.template = template
        self.template_inv = 1-self.template
        self.front = front
        self.front_inv = 1-self.front
        self.back = back
        self.back_inv = 1-self.back
        self.coarse_template, self.coarse_front, self.coarse_back\
            = self.make_coarse(template,front,back)

    def make_coarse(self,length_factor=2,edge_threshold=.7):
        full_template = np.hstack((self.front[:,:self.front_length/2],
                                   self.template,
                                   self.back[:,self.back_length:]))
        self.coarse_full_template = length_resize(full_template,length_factor)
        # threshold the values to only get a few indices
        self.template_bool = self.coarse_full_template > .7
        self.coarse_template = self.coarse_full_template[self.template_bool]

    def get_deformed_template(E,base_loc,front_displace,back_displace):
        """ Given an edge feature map and a location find
        the best spot, ideal should be a dynamic programming
        algorithm, for now we keep the middle template
        fixed and we just put the other two templates
        where-ever in order to get a segment that defines
        where the occurrence is
        """
        # get a deformed model
        deformed_length = self.back_start+self.back_length-self.front_start
        deformed_template = np.zeros((self.height,deformed_length))
        deformed_template[:,0:self.front_length] = self.front.copy()
        deformed_template[:,-self.back_length:] = self.back.copy()
        new_middle_start = self.middle_start-front_displace
        new_back_start = self.back_start-front_displace+back_displace
        if new_middle_start < self.front_length:
            front_overlap = self.front_length-new_middle_start
            deformed_template[:,new_middle_start:self.front_length] = (deformed_template[:,new_middle_start:self.front_length] + self.template[:,:front_overlap])/2.
            middle_sec_start = self.front_length
            middle_start_idx = front_overlap
        else:
            middle_sec_start = new_middle_start
            middle_start_idx = 0
        if new_back_start < middle_sec_start+self.length:
            middle_sec_end = new_back_start
            middle_end_idx = middle_start_idx + new_back_start-middle_sec_start
        else:
            middle_sec_end = self.length+middle_sec_start
            middle_end_idx = middle_start_idx+self.length
        deformed_template[:,middle_sec_start:middle_sec_end]= self.template[:,middle_start_idx:middle_end_idx].copy()
        assert(middle_sec_end<=new_back_start)
        # overlap between back template and middle template
        if middle_end_idx < self.length:
            back_overlap = self.length-middle_end_idx
            assert(middle_sec_end) == new_back_start
            deformed_template[:,new_back_start:new_back_start+back_overlap] = (deformed_template[:,new_back_start:new_back_start+back_overlap]+self.template[:,middle_ned_idx:])/2.
        return deformed_template


    def patchwork_likelihood(E,base_loc,front_displace,back_displace):
        pass
        
        
        

def length_resize(I,length_factor=2):
    height,length = I.shape
    new_length = length/length_factor
    length_resize_mat = np.inf * np.ones((length_factor,height,new_length))
    for l in xrange(length_factor):
        sec_length = (length-l)/length_factor
        length_resize_mat[l,:,:sec_length] = I[:,length_factor*np.arange(sec_length)+l]
    return np.max(length_resize_mat,axis=0)

def estimate_fronts_backs(all_fronts_backs):
    """ Take an average of the fronts and backs
    Parameters:
    ===========
    all_fronts_backs:
        List of the front and back tuples

    Returns:
    ========
    all_fronts:
        ndarray of the front estimated template
    all_backs:
        ndarray of the back estimated template
    """
    all_fronts = np.array([fb[0] for fb in all_fronts_backs])
    all_backs = np.array([fb[1] for fb in all_fronts_backs])
    return np.maximum(np.minimum(np.mean(all_fronts,\
                                             axis=0),\
                                     .95),\
                          .05),\
                          np.maximum(np.minimum(np.mean(all_backs,\
                                                            axis=0),\
                                                    .95),\
                                         .05)

def get_front_back_fit(part_triple):
    """ Fits a part model (a triple with template, front and back)
    
    """
    
