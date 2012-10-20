#
# TODO: load in an E example
#       create a file that has all relevant functions for waliji feature processing
#       check by hand that patches with too few edges do in fact get excluded
#       check by hand that we are validly computing bernoulli likelihoods

import waliji_base as wb

def get_part_scores(E,log_part_blocks,log_invpart_blocks,
                    frequency_mode='valid',time_mode='same'):
    part_block_shape = log_part_blocks[0].shape
    if time_mode=='valid':
        t_start = part_block_shape[0]/2
        t_end = t_start + E.shape[0] - part_block_shape[0] + 1
    else: # feature_mode == 'same'
        # TODO: check whether I should make the modes available
        #       type-safe
        t_start = 0
        t_end = E.shape[0]
    # check the frequency parameters
    if frequency_mode=='valid':
        f_start = part_block_shape[1]/2
        f_end = f_start + E.shape[1] - part_block_shape[1] + 1
    else: # frequency_mode == 'same'
        f_start = 0
        f_end = E.shape[1]
    e_pos = part_block_shape[2]/2
    return np.array([
            (ndimage.correlate(E,log_part_block)[t_start:t_end,
                                                 f_start:f_end,
                                                 e_pos]
             +ndimage.correlate(1-E,log_invpart_block)[t_start:t_end,
                                                       f_start:f_end,
                                                       e_pos])
            for log_part_block,log_invpart_block in zip(log_part_blocks,
                                                        log_invpart_blocks)])


def code_parts(E,log_part_blocks,log_invpart_blocks,edge_count_lower_bound=30,
               frequency_mode='valid',time_mode='same'):
    """
    Assumes that E has been reorganized to have
    dimension 0 be the time axis
    dimension 1 be the frequency axis
    dimension 2 be the edge type axis
    
    Parameters:
    ===========
    log_part_blocks: ndarray[ndim=4,dtype=float]        
        Dimension 0 is the part index, dimension 1 is a time index
        dimension 2 is a frequency index, dimension 3 is an edge type
        index.  These are log probabilities over the occurance of an
        edge of the particular type at those relative time and
        frequency locations

    log_invpart_blocks: ndarray[ndim=4,dtype=float]
        Same information as the array in log_part_blocks, assumed to be
        precomputed for time saving this is equal to:
        
                   np.log( 1. - np.exp( log_part_blocks))
                   
        hence this allows for fast bernoulli model computation.

    Output:
    =======
    Part_Index_Map: ndarray[ndim=2,dtype=int]
        This has only two dimensions and those dimensions are equal
        to the first two dimensions of E. The values of these entries
        are the part indices.  

    
    """
    part_scores = get_part_scores(
        E,
        np.vstack((np.ones((1,)+log_part_blocks.shape[1:]),
                   log_part_blocks)),
        np.vstack((np.zeros((1,)+log_part_blocks.shape[1:]),
                   log_invpart_blocks)),
        frequency_mode=frequency_mode,time_mode=time_mode)
    # get upper bound on the part scores
    part_score_upper_bound_strict = E.max()+1.
    # find the significant counts
    part_scores[0][part_scores[0] < edge_count_lower_bound] = part_score_upper_bound_strict
    part_scores[0][part_scores[0] < edge_count_lower_bound] = part_score_upper_bound_strict
    return np.argmax(
        part_scores,0)


F = wb.code_parts(wb.E,wb.log_part_blocks,wb.log_invpart_blocks,edge_count_lower_bound=30)

frequency_mode='valid'
time_mode='same'

part_scores = wb.get_part_scores(
        wb.E,
        np.vstack((np.ones((1,)+wb.log_part_blocks.shape[1:]),
                   wb.log_part_blocks)),
        np.vstack((np.zeros((1,)+wb.log_part_blocks.shape[1:]),
                   wb.log_invpart_blocks)),
        frequency_mode=frequency_mode,time_mode=time_mode)

part_score_upper_bound_strict = part_scores.max()+1.
part_score_lower_bound_strict = part_scores.min()-1.
# find the significant counts
edge_count_lower_bound = 10
part_scores[0][part_scores[0] < edge_count_lower_bound] = part_score_upper_bound_strict
part_scores[0][part_scores[0] >= edge_count_lower_bound] = part_score_lower_bound_strict
