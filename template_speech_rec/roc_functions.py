import numpy as np
import itertools
import cluster_times

def get_auto_syllable_window(template):
    return int( -np.ceil(template.shape[0]/3.)), - int(np.ceil(template.shape[0]/3.))

def get_C0_C1(template):
    return template.shape[0], int(np.ceil(template.shape[0]*1.5))

def get_max_detection_in_syllable_windows(detection_array,
                                          example_start_end_times,
                                          detection_lengths,
                                          window_start,
                                          window_end):
    """
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
    return cluster_tuples

def perform_detection_time_clustering(potential_detect_times,C0,C1):
    """
    Assume that potential_detect_times needs to be an array of
    np.uint16 entries that is monotonically strictly increasing
    """
    cluster_partition = cluster_times.cluster_times(potential_detect_times,
                                                    C0,
                                                    C1)
    return get_cluster_starts_ends(potential_detect_times,cluster_partition)

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

def true_false_positive_rate(threshold,
                        detection_array,
                        detection_lengths,
                        example_detection_windows,
                        C0,C1,frame_rate):
    num_frames = detection_lengths.sum()
    cluster_starts_ends = tuple(
        perform_detection_time_clustering(
            get_max_above_threshold(detect_vals,
                                    detect_length,
                                    threshold,C0),
            C0,C1)
        for detect_vals, detect_length in itertools.izip(detection_array,
                                                         detection_lengths))
    return get_rates(np.array(tuple(
            true_false_clusters(example_cluster_starts_ends,
                                example_start_end_tuples)
            for example_cluster_starts_ends, example_start_end_tuples in itertools.izip(cluster_starts_ends,example_detection_windows))).T,
                     num_frames,
                     frame_rate)



def get_roc_curve(potential_thresholds,
                  detection_array,
                  detection_lengths,
                  example_start_end_times,
                  C0,C1,frame_rate,
                  win_front_length = None,
                  win_back_length = None):
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
    return rate_mat[0], rate_mat[1]


def display_roc_curve(file_path,
                      false_positive_rate,
                      true_positive_rate):
    pass

#######################################
#  Binary search functions 
#
#
########################################
"""
Purpose of this section is to be able to find the threshold which gives
the largest true positive rate while bounding the false positive
rate from above.  This gives us an initial function:

find_best_true_positive_bound_false_positive
"""

def find_best_true_positive_bound_false_positive(detection_array,
                                                 threshold_candidates,
                                                 detection_lengths,
                                                 false_positive_rate):
    """
    Parameters:
    ===========
    detection_array: np.ndarray[ndim=2,]
        We will verify and set it to dtype np.float32 in this function.
        Contains all the detections for the utterances that are processed
    threshold_candidates: np.ndarray[ndim=1,dtype=np.float32]
        These are all the thresholds that could be used, essentially
        for each true example in the set of test utterances, there is
        a threshold that guarantees that that particular example
        will be detected.  These are the thresholds corresponding
        to those values.  We assume these are sorted from least to greatest
    detection_lengths: np.ndarray[ndim=1,dtype=int]
        detection_lengths[i] is the number of frames in utterance i

    Output:
    =======
    false_positive_rates:
    threshold_boundary_cmps
    """
    num_thresholds = threshold_candidates.shape[0]
    false_positive_rates = np.zeros(num_thresholds)
    threshold_boundary_cmps = np.zeros((num_thresholds+1,2))
    

    
                                                 
