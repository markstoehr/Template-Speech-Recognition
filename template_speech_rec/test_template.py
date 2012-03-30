import numpy as np
from scipy.signal import convolve

def get_utterance_score(T,U,bg_len,maxima_radius=3):
    P,C,start_idx,end_idx = _apply_template_to_utterance(T,U,bg_len)
    scores = P+C
    scores_maxima = _get_maxima(scores,maxima_radius)
    start_idx = start_idx + maxima_radius
    end_idx = end_idx - maxima_radius
    return scores[maxima_radius:-maxima_radius], scores_maxima,start_idx,end_idx


def score_template_background_section(T,bgd,E):
    # check that the length of the example utterance
    # is the same as the template length
    assert (E.shape[1] == T.shape[1])
    template_length = T.shape[1]
    U_bgd = np.tile(bgd,(template_length,1)).transpose()
    T_inv = 1 - T
    U_bgd_inv = 1 - U_bgd
    C_exp_inv_long = T_inv/U_bgd_inv
    # get the C_k
    C = np.log(C_exp_inv_long).sum()
    return (E*np.log(T/U_bgd / C_exp_inv_long)).sum(),C 


def cmp_score_to_label(scores,scores_maxima,start_idx,pattern_times,detect_radius):
    pass


def _apply_template_to_utterance(T,U,bg_len):
    """ Returns an array of detection scores
    """
    num_detections = U.shape[0]-T.shape[0]+1
    U_background,start_idx,end_idx = _get_background(U,bg_len)
    U_stack = np.tile(U[:,start_idx:end_idx],(T.shape[1],1))
    template_length = T.shape[1]
    U_bg_stack = np.tile(U_background,(template_length,1))
    T_stack = np.tile(T.transpose().reshape(np.prod(T.shape),1),(1,U_background.shape[1]))
    T_stack_inv = 1 - T_stack
    U_bg_stack_inv = 1 - U_bg_stack
    C_exp_inv_long = T_stack_inv/U_bg_stack_inv
    # get the C_k
    C = np.log(C_exp_inv_long).sum(0)
    return (U_stack*np.log(T_stack/U_bg_stack / C_exp_inv_long)).sum(0),C,start_idx,end_idx
    
def _get_maxima(scores,maxima_radius):
    num_potential_max = scores.shape[0] - 2*maxima_radius
    potential_max_idx = np.arange(maxima_radius+1,num_potential_max+maxima_radius+1)
    score_cmp_stack = np.tile(np.arange(num_potential_max),
            (2*maxima_radius+1,1)) \
            + np.arange(2*maxima_radius+1).reshape(2*maxima_radius+1,1)
    return np.logical_and(scores[potential_max_idx]>=scores[score_cmp_stack].max(0),
                          scores[potential_max_idx]>scores[score_cmp_stack].min(0))
    
    
    
    

def _get_background(U,bg_len):
    """ Returns the background adaptively computed
    also the starting and ending indices so that
    we can slice the section of the utterance
    and the labels which correspond to the background
    that was selected
    background is truncated to be in [.05,.4]
    """
    return np.minimum(np.maximum(convolve(U.transpose(), 
             1./bg_len*np.ones((bg_len,1)),
             "valid").transpose(),.05),.4),\
             int(np.floor(bg_len/2.)),\
             int(np.floor(bg_len/2.)) + U.shape[1]-bg_len+1
