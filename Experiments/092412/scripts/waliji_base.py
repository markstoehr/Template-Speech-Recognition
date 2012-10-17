import numpy as np
from scipy import ndimage

import template_speech_rec.extract_local_features as elf

#
# create function to adjust parts into a useable form for filtering
#
def reorg_part_for_fast_filtering(part,feature_types=8):
    """
    Assumes the patch for different edge types have been vertically stacked
    and that there are eight edge types
    dimensions are features by time
    want time by feature by edge type
    """
    H = part.shape[0]/feature_types
    return np.array([
            part[i*H:(i+1)*H].T
            for i in xrange(feature_types)]).swapaxes(0,1).swapaxes(1,2)
            
def reorg_parts_for_fast_filtering(parts,feature_types=8):
    return np.array([
            reorg_part_for_fast_filtering(part,feature_types=feature_types)
            for part in parts])



def collapse_to_grid(E_coded,grid_time,grid_frequency,num_codes,
                     do_grid_subsampling =True):
    """
    Parameters:
    ===========
    E_coded: numpy.ndarray[ndim=2,dtype=int]
        Feature map that indicates the presence of the waliji feature
    grid_time: int
    grid_frequency: int
    """
    full_grid = np.dstack(tuple(
            generic_filter(E_coded,lambda x: np.any(x==i).astype(np.uint8),
                           size = (grid_time,
                                   grid_frequency))
            for i in xrange(num_codes)))
    if do_grid_subsampling:
        return full_grid[::grid_time,::grid_frequency]
    else:
        return full_grid

def test_collapsed_grid(collapsed_grid,E_coded,
                        grid_time,grid_frequency):
    assert E_coded.max() <= collapsed_grid.shape[-1]
    assert E_coded.min() >= 0
    



import code_parts as cp

code_parts_fast = cp.code_parts_fast
'''

'''

F2 = cp.code_parts_fast(E.astype(np.uint8),log_part_blocks,log_invpart_blocks,20)

F = code_parts(E,log_part_blocks,log_invpart_blocks,
               edge_count_lower_bound=20)
F = collapse_to_grid(F,log_part_blocks.shape[1],
                     log_part_blocks.shape[2],
                     log_part_blocks.shape[0])



