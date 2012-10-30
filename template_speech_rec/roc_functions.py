import numpy as np
import itertools
import cluster_times
from scipy.stats import gaussian_kde

def get_auto_syllable_window(template):
    return int( -np.ceil(template.shape[0]/3.)), int(np.ceil(template.shape[0]/3.))

def get_auto_syllable_window_mixture(templates):
    tshape = max(tuple(t.shape[0] for t in templates))
    return int( -np.ceil(tshape/3.)), int(np.ceil(tshape/3.))

def get_C0_C1(template):
    return template.shape[0], int(np.ceil(template.shape[0]*1.5))

def get_C0_C1_mixture(templates):
    tshape = max(tuple(t.shape[0] for t in templates))
    return tshape, int(np.ceil(tshape*1.5))

def compute_fom(true_positive_rates, false_positive_rates):
    main_detect_area = (false_positive_rates <= 10./360) * (false_positive_rates >= 1./360)
    return true_positive_rates[main_detect_area].sum()/main_detect_area.sum()

def get_max_detection_in_syllable_windows(detection_array,
                                          example_start_end_times,
                                          detection_lengths,
                                          window_start,
                                          window_end):
    """
    The max detect vals essentially returns the potential thresholds
    for detecting examples.  In particular for a given instance in the
    data we have detection windows associated with that instance where
    any detection that is within the window makes sense. For a given
    instance, the maximum value of the response curve within the
    detection window is the smallest threshold where we will have a
    positive detection of the instance.

    Hence the max_detect_vals correspond to candidates for the
    thresholds.

    Parameters:
    ===========
    aar_template:
       included to set the lengths of the detection windows adaptively
    window_start: int
        window_start <= 0, it says at what frame does the beginning of the detection window start, default is 1/3 of template length
    window_end: int
        at what from does the detection window end from the beginning
        of the syllable
        default is 1/3 of the template length
    """
    max_detect_vals = []
    for example_idx,start_end_times in enumerate(example_start_end_times):
        for start_time, end_time in start_end_times:
            print start_time
            max_detect_vals.append(detection_array[example_idx,max(start_time+window_start,0):
                                                       min(start_time+window_end,detection_array.shape[1])].max())
    return np.sort(max_detect_vals)


def get_max_above_threshold(detection_vals,detection_length,threshold,C0):
    l = detection_length-C0+1
    return np.arange(1,l-1,dtype=np.uint16)[
        (detection_vals[1:l-1] >= detection_vals[:l-2])
        * (detection_vals[1:l-1] >= detection_vals[2:l])
        * (detection_vals[1:l-1] >= threshold)]

def get_cluster_starts_ends(potential_detect_times,cluster_partition):
    """
    Assumptions:
    len(potential_detect_times) == len(cluster_partition)-1
    set(list(potential_detect_times)) =[0,2]
    len(cluster_partition) >= 2
    cluster_partition[0] == 2
    cluster_partition[-1] == 2

    The output is cluster_tuples
    cluster_tuples
    """
    d = len(potential_detect_times)
    t = 0
    cluster_tuples = []
    while t < d:
        cur_cluster_start = potential_detect_times[t]
        t += 1
        while cluster_partition[t] != 2:
            t += 1
        cur_cluster_end = potential_detect_times[t-1]+1
        cluster_tuples.append((cur_cluster_start,
                               cur_cluster_end))
    return tuple(cluster_tuples)

def perform_detection_time_clustering(potential_detect_times,C0,C1):
    """
    Assume that potential_detect_times needs to be an array of
    np.uint16 entries that is monotonically strictly increasing
    """
    cluster_partition = cluster_times.cluster_times(potential_detect_times,
                                                    C0,
                                                    C1)
    return get_cluster_starts_ends(potential_detect_times,cluster_partition)

def true_false_clusters_list(example_cluster_starts_ends,
                        example_start_end_tuples):
    num_true = len(example_start_end_tuples)
    example_list = np.zeros(num_true,dtype=np.uint8)
    num_detects = len(example_cluster_starts_ends)
    true_detects = 0
    false_detects = 0
    if num_true == 0:
        # num_true is zero penalize all detections
        return example_list
    elif num_detects == 0:
        return example_list
    else:
        true_iter = enumerate(example_start_end_tuples)
        true_idx, cur_true = true_iter.next()
        for i,cur_detect in enumerate(example_cluster_starts_ends):
            # check whether the current detection cluster is
            # passed the marked time point
            while cur_detect[0] >= cur_true[1]:
                if true_idx == num_true -1:
                    # we are done now since there are no more true
                    # detections possible
                    return example_list
                else:
                    true_idx, cur_true = true_iter.next()
            # we now know that cur_detect[0] < cur_true[1]
            # we have a detection if cur_detect[1] > cur_true[0]
            example_list[true_idx] = np.uint8(cur_detect[1] > cur_true[0])


    return example_list


def true_false_clusters(example_cluster_starts_ends,
                        example_start_end_tuples):
    num_true = len(example_start_end_tuples)
    num_detects = len(example_cluster_starts_ends)
    true_detects = 0
    false_detects = 0
    if num_true == 0:
        # num_true is zero penalize all detections
        return num_detects, 0, num_true
    elif num_detects == 0:
        return 0, 0, num_true
    else:
        true_iter = enumerate(example_start_end_tuples)
        true_idx, cur_true = true_iter.next()
        for i,cur_detect in enumerate(example_cluster_starts_ends):
            # check whether the current detection cluster is
            # passed the marked time point
            while cur_detect[0] >= cur_true[1]:
                if true_idx == num_true -1:
                    # we are done now since there are no more true
                    # detections possible
                    return num_detects - true_detects, true_detects, num_true
                else:
                    true_idx, cur_true = true_iter.next()
            # we now know that cur_detect[0] < cur_true[1]
            # we have a detection if cur_detect[1] > cur_true[0]
            true_detects += cur_detect[1] > cur_true[0]
    return num_detects - true_detects, true_detects, num_true

def get_rates(true_false_clusters_array,num_frames,frame_rate):
    """
    Input is a 2-d array with N columns where N is the number
    of utterances.  There are three rows so
    each column has three entries, the first is the number of false positives
    the second is the number of true positives, and the third is the number
    of true examples within the utterance
    """
    false_positives, true_positives, total_true = tuple(true_false_clusters_array.sum(1))
    return (false_positives /float(num_frames) * frame_rate,
            true_positives/float(total_true))


def generate_example_detection_windows(example_start_end_times,
                                       win_front_length,
                                       win_back_length):
    return tuple(
        tuple( (s -win_front_length, s+win_back_length)
               for s,e in utt_examples)
        for utt_examples in example_start_end_times)

def _get_detect_clusters_single_threshold(threshold,
                                          detection_array,
                                          detection_lengths,
                                          C0,C1):
    """
    a clustering is dependent upon a threshold level the treshold
    levels are the max_detect_vals for each value in max_detect_vals
    there is going to be a cluster these clusters are determined by
    the detection array, and the parameters for the shapes of the
    clusters the output for a given threshold should be a list of
    segments for each utterance in the rows of detection_array

    Parameters:
    ===========
    threshold: float
        detection threshold
    """
    return tuple(
        perform_detection_time_clustering(
            get_max_above_threshold(detect_vals,
                                    detect_length,
                                    threshold,C0),
            C0,C1)
        for detect_vals, detect_length in itertools.izip(detection_array,
                                                         detection_lengths))

def get_detect_clusters_threshold_array(thresholds,
                                        detection_array,
                                        detection_lengths,
                                        C0,C1):
    return tuple(
        _get_detect_clusters_single_threshold(threshold,
                                          detection_array,
                                          detection_lengths,
                                          C0,C1)
        for threshold in thresholds)

def true_false_positive_rate(threshold,
                        detection_array,
                        detection_lengths,
                        example_detection_windows,
                        C0,C1,frame_rate,
                        return_list = False,
                        return_clusters=False):
    num_frames = detection_lengths.sum()
    cluster_starts_ends = tuple(
        perform_detection_time_clustering(
            get_max_above_threshold(detect_vals,
                                    detect_length,
                                    threshold,C0),
            C0,C1)
        for detect_vals, detect_length in itertools.izip(detection_array,
                                                         detection_lengths))
    if return_clusters:
        return cluster_starts_ends
    if not return_list:
        return get_rates(np.array(tuple(
            true_false_clusters(example_cluster_starts_ends,
                                example_start_end_tuples)
                                for example_cluster_starts_ends, example_start_end_tuples in itertools.izip(cluster_starts_ends,example_detection_windows))).T,
                                num_frames,
                                frame_rate)
    else:
        return tuple(
            true_false_clusters(example_cluster_starts_ends,
                                example_start_end_tuples)
                                for example_cluster_starts_ends, example_start_end_tuples in itertools.izip(cluster_starts_ends,example_detection_windows))


def get_segment_phn_context(l,
                            phns,
                            phn_id_maps,
                            phn_half_win_size):
    return tuple(
        phns[s-phn_half_win_size:
                 e+phn_half_win_size+1]
        for s,e in l)

def get_phn_id_maps(flts):
    return np.array(tuple( phn_id
                           for phn_id,flt in enumerate(flts[1:] - flts[:-1])
                           for i in xrange(flt)))

def return_detector_output(true_detection_list,
                           false_detection_list,
                           false_negative_list,
                           phns,
                           flts,
                           phn_half_win_size,
                           return_context,
                           return_lists):
    """
    Organizes which output to give
    """
    out_lists = (true_detection_list,
                 false_detection_list,
                 false_negative_list)
    return_tuple = tuple(len(l) for l in out_lists)
    if return_lists:
        return_tuple += out_lists
    if return_context:
        phn_id_maps = get_phn_id_maps(phns,flts)
        return_tuple += tuple(
            get_segment_phn_context(l,
                                    phns,
                                    phn_id_maps,
                                    phn_half_win_size))
        
    return return_tuple
    
    




def get_detection_rates_from_clusters(true_clusters,
                                      detected_clusters,
                                      phns=None,
                                      flts=None,
                                      phn_half_win_size=2,
                                      return_context=False,
                                      return_lists=False):
    """
    Output the number of false positives and the number of true positives.

    detected_clusters is a list of tuples (representing a half open
    interval) where we propose that a detection has occurred.
    true_clusters is a list of tuples (also representing half open intervals)
    where we expect the auditory object occurred.

    The false positives is the set of the detected_clusters which do not
    overlap with any of the true segments.

    The true positives are the set of tuples in true_clusters which
    overlap with an interval in detected_clusters.

    We note that multiple detections are not penalized under this system.

    We also have an optional system for checking what the phonetic
    context is for true detections and false detections

    The output depends on whether return_context is true or false.
    If ``return_context == False `` then we only return the number
    of true detections and false positives.
    If ``return_context == True `` then we return, in addition to
    the statistics, also the list of phone contexts for each true detection,
    each false positive, and each false negative.
    """
    if return_context:
        assert (phns is not None) and (flts is not None)

    num_true = len(true_clusters)
    num_detected = len(detected_clusters)
    true_detection_list = []
    false_detection_list = []
    false_negative_list = []

    if num_true == 0:
        false_detection_list =detected_clusters
        return return_detector_output(true_detection_list,
                                      false_detection_list,
                                      false_negative_list,
                                      phns,
                                      flts,
                                      phn_half_win_size,
                                      return_context,
                                      return_lists)
    elif num_detected == 0:
        false_negative_list = true_clusters
        return return_detector_output(true_detection_list,
                                      false_detection_list,
                                      false_negative_list,
                                      phns,
                                      flts,
                                      phn_half_win_size,
                                      return_context,
                                      return_lists)
    else: # assume here that we have both true clusters and detected clusters
        pass

    true_detects = np.zeros(num_true,dtype=bool)
    false_detects = np.ones(num_detected,dtype=bool)
    for true_cluster_id, true_cluster in enumerate(true_clusters):
        for detected_cluster_id, detected_cluster in enumerate(detected_clusters):
            # check if there is an overlap
            if true_cluster[1] > detected_cluster[0] and true_cluster[0] < detected_cluster[1]:
                true_detects[true_cluster_id] = True
                false_detects[detected_cluster_id] = False
            else: # no overlap, false negative
                false_negative_list.append(true_cluster)
    
    return return_detector_output([true_clusters[i] for i,v in enumerate(true_detects) if v],
                                  [detected_clusters[i] for i,v in enumerate(false_detects) if v],
                                  false_negative_list,
                                  phns,
                                  flts,
                                  phn_half_win_size,
                                  return_context,
                                  return_lists)

        

def get_detection_rates_from_clusters_utterance(seq_of_true_segments,
                                                seq_of_detected_clusters):
    """
    Returns the false positive rate and the false negative rate
    """
    stats_rates = np.array([
            get_detection_rates_from_clusters(true_segment,detected_clusters)
            for true_segment, detected_clusters in itertools.izip(seq_of_true_segments,
                                      seq_of_detected_clusters)]).T
    return stats_rates[1],stats_rates[2]

def get_roc_curve(potential_thresholds,
                  detection_array,
                  detection_lengths,
                  example_start_end_times,
                  C0,C1,frame_rate,
                  win_front_length = None,
                  win_back_length = None,
                  return_detected_examples=False,
                  return_clusters = False):
    if win_front_length is None:
        win_front_length = int(np.ceil(C0/3.))
    if win_back_length is None:
        win_back_length = int(np.ceil(C0/3.))
    example_detection_windows = generate_example_detection_windows(example_start_end_times,
                                       win_front_length,
                                       win_back_length)
    rate_mat = np.array([
            true_false_positive_rate(threshold,
                                     detection_array,
                                     detection_lengths,
                                     example_detection_windows,
                                     C0,C1,frame_rate)
            for threshold in potential_thresholds]).T
    return_tuple = (rate_mat[0],rate_mat[1])
    if return_detected_examples:
        return_tuple +=  (true_false_positive_rate(threshold,
                                                     detection_array,
                                                     detection_lengths,
                                                     example_detection_windows,
                                                     C0,C1,frame_rate,
                                                     return_list=True),)
    if return_clusters:
        return_tuple += (true_false_positive_rate(threshold,
                                                     detection_array,
                                                     detection_lengths,
                                                     example_detection_windows,
                                                     C0,C1,frame_rate,
                                                     return_clusters=True),)
        return rate_mat[0], rate_mat[1]
    return return_tuple


def display_roc_curve(file_path,
                      false_positive_rate,
                      true_positive_rate):
    pass

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
