import numpy as np
import bernoulli_em as bem
import edge_signal_proc as esp

def extract_local_features_tied(E,patch_height,patch_width,
                                lower_cutoff,upper_cutoff,
                                edge_feature_row_breaks):
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
    all_patch_rows = np.zeros(0,dtype=np.uint16)
    all_patch_cols = np.zeros(0,dtype=np.uint16)
    patch_row_ids, patch_col_ids = get_flat_patches2E_indices(E[edge_feature_row_breaks[0]:
                                                                edge_feature_row_breaks[1]],
                                                            patch_height,patch_width)
    bp_tmp = _extract_block_local_features_tied(
                        E[edge_feature_row_breaks[0]:
                              edge_feature_row_breaks[1]],
                        patch_height,patch_width)
    for edge_id in xrange(1,edge_feature_row_breaks.shape[0]-1):
        bp_tmp = np.hstack((bp_tmp,
                            _extract_block_local_features_tied(
        E[edge_feature_row_breaks[edge_id]:
          edge_feature_row_breaks[edge_id+1]],
          patch_height,patch_width)))
        use_indices = np.argsort(np.sum(np.sum(bp_tmp,axis=1),axis=1))[lower_quantile*bp_tmp.shape[0]:
                                                                           upper_quantile*bp_tmp.shape[0]]
        all_patch_rows = np.hstack((all_patch_rows,patch_row_ids[use_indices]))
        all_patch_cols = np.hstack((all_patch_cols,patch_col_ids[use_indices]))
        bp_tmp=bp_tmp[use_indices]
        bp = np.vstack((bp,bp_tmp))

    return bp,all_patch_rows,all_patch_cols

def extract_local_features_tied_segments(E,patch_height,patch_width,
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
    all_patch_rows = np.zeros(0,dtype=np.uint16)
    all_patch_cols = np.zeros(0,dtype=np.uint16)
    num_segs = E.shape[1]/segment_length
    for segment_id in xrange(num_segs-1):
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
    # finish of the last segment
    segment_id = num_segs-1
    patch_row_ids, patch_col_ids = get_flat_patches2E_indices(E[edge_feature_row_breaks[0]:
                                                                edge_feature_row_breaks[1],
                                                                segment_id*segment_length:],
                                                                patch_height,patch_width)
    patch_col_ids += segment_id*segment_length
    bp_tmp = _extract_block_local_features_tied(
        E[edge_feature_row_breaks[0]:
          edge_feature_row_breaks[1],
          segment_id*segment_length:],
                        patch_height,patch_width)
    for edge_id in xrange(1,edge_feature_row_breaks.shape[0]-1):
        bp_tmp = np.hstack((bp_tmp,
                            _extract_block_local_features_tied(
        E[edge_feature_row_breaks[edge_id]:
          edge_feature_row_breaks[edge_id+1],
          segment_id*segment_length:],
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
    patch_row_ids = np.arange(num_patches,dtype=np.uint16) / num_patches_across
    patch_col_ids = np.arange(num_patches,dtype=np.uint16) % num_patches_across
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

def generate_patch_row_indices(patch_rows,patch_height,patch_width):
    patch_size = patch_height*patch_width
    base_patch_row_indices = np.tile(np.arange(patch_height),(patch_width,1)).T.reshape(patch_size)
    base_patch_row_rep = np.tile(base_patch_row_indices,patch_rows.shape[0])
    patch_row_rep = np.tile(patch_rows,(patch_size,1)).T.reshape(base_patch_row_rep.shape[0])
    return base_patch_row_rep + patch_row_rep

def generate_patch_col_indices(patch_cols,patch_height,patch_width):
    patch_size = patch_height*patch_width
    base_patch_col_indices = np.tile(np.arange(patch_width),(patch_height,1)).reshape(patch_size)
    base_patch_col_rep = np.tile(base_patch_col_indices,patch_cols.shape[0])
    patch_col_rep = np.tile(patch_cols,(patch_size,1)).T.reshape(base_patch_col_rep.shape[0])
    return base_patch_col_rep + patch_col_rep


def patch_col_idx_to_s(patch_col_idx,patch_width,data_iter):
    start_idx = patch_col_idx*data_iter.num_window_step_samples
    end_idx = (patch_col_idx+(patch_width-1))*data_iter.num_window_step_samples + data_iter.num_window_samples
    return data_iter.s[start_idx:end_idx]

def patch_col_indices_to_s(all_patch_cols,patch_width,data_iter,offset = 80):
    """
    Returns matrix with a row for each entry in patch_col_indices
    and each row corresponds to the window of s
    """
    base_patch_window = np.arange( (patch_width-1)*data_iter.num_window_step_samples+data_iter.num_window_samples)
    windows = np.tile(all_patch_cols,(base_patch_window.shape[0],1)).T \
        + np.tile(base_patch_window,(all_patch_cols.shape[0],1))
    return data_iter.s[windows.reshape(base_patch_window.shape[0] * all_patch_cols.shape[0])].reshape(all_patch_cols.shape[0],base_patch_window.shape[0])


def local_features_blocks(data_iter,num_iter,
                      patch_height,patch_width,
                      lower_quantile,upper_quantile,
                      num_edge_features=8,segment_ms=500,
                      hop_ms = 5):
    """
    Get the binary edge patters associated that have a lot of edges
    also extract the corresponding spectrogram pattern
    we also extract the chunks of the raw signal that correspond to the patterns
    """
    bps = np.zeros((0,num_edge_features*patch_height,patch_width))
    spec_patches = np.zeros((0,patch_height,patch_width))
    signal_windows = np.zeros((0,patch_col_indices_to_s(np.array([1]),patch_width,data_iter).shape[1]))
    patch_rows = np.zeros(0)
    patch_cols = np.zeros(0)
    for k in xrange(num_iter):
        if not data_iter.next():
            return None
        E, edge_feature_row_breaks, edge_orientations =\
            data_iter.E, data_iter.edge_feature_row_breaks, data_iter.edge_orientations
        S = data_iter.S
        bp,all_patch_rows,all_patch_cols = extract_local_features_tied(E,patch_height,
                                                                      patch_width, lower_quantile,
                                                                      upper_quantile,
                                                                      edge_feature_row_breaks,
                                                                      segment_ms=500,
                                                                      hop_ms = 5)
        bps = np.vstack((bps,bp))
        patch_row_indices = generate_patch_row_indices(all_patch_rows,patch_height,patch_width)
        patch_col_indices = generate_patch_col_indices(all_patch_cols,patch_height,patch_width)
        signal_windows = np.vstack((signal_windows,patch_col_indices_to_s(all_patch_cols,patch_width,data_iter)))
        spec_patches = np.vstack((spec_patches,
                                  S[patch_row_indices,patch_col_indices].reshape(bp.shape[0],patch_height,patch_width)))
    return bps, spec_patches, signal_windows

