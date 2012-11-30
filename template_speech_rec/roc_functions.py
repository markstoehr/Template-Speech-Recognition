import numpy as np
import itertools
import cluster_times
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import collections
import get_train_data as gtrd

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
                                          window_end,
                                          verbose=False):
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
            detect_vals = detection_array[example_idx,max(start_time+window_start,0):
                                      min(start_time+window_end,detection_array.shape[1])]
            maybe_val_array = detect_vals[1:-1][(detect_vals[1:-1] >= detect_vals[:-2])
                                    * (detect_vals[1:-1] >= detect_vals[2:])]
            if len(maybe_val_array) > 0:
                val = maybe_val_array.max()
            else:
                val = -np.inf
            if verbose:
                print "Utt: %d, loc: %d, val:%g" % (example_idx,start_time,val)
            max_detect_vals.append(val)
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
                        example_start_end_tuples,
                        verbose =False):
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
            if verbose:
                print i, cur_detect
            # check whether the current detection cluster is
            # passed the marked time point
            while cur_detect[0] >= cur_true[1]:
                if verbose:
                    print "cur_detect[0]=%d,cur_true=%d" % (cur_detect[0], cur_true[0])
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
        if threshold > - np.inf
        else ()
        for threshold in thresholds )

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

    
def get_pos_neg_detections(detection_clusters_at_threshold,
                           detection_array,
                           C1,
                           window_start,
                           window_end,
                           example_start_end_times):
    """
    Parameters:
    ===========
    detection_clusters_at_threshold:
    detection_array:
    C1:
    window_start:
    window_end:
    example_start_end_times:

    Output:
    =======
    pos_clusters
    neg_clusters
    """
    num_clusters = sum( len(cset) for cset in detection_clusters_at_threshold)
    num_pos_clusters = 0
    num_neg_clusters = 0
    pos_clusters = np.zeros((num_clusters,C1))
    neg_clusters = np.zeros((num_clusters,C1))
    for detect_clusters, detection_row, start_end_times in itertools.izip(detection_clusters_at_threshold,detection_array,example_start_end_times):
        for c in detect_clusters:
            is_neg = True
            for s,e in start_end_times:
                if s+window_start <= c[1] and s+window_end >= c[0]:
                    is_neg = False
                    pos_clusters[num_pos_clusters] = get_threshold_neighborhood(c,detection_row,C1)
                    num_pos_clusters += 1
            if is_neg:
                neg_clusters[num_neg_clusters] = get_threshold_neighborhood(c,detection_row,C1)
                num_neg_clusters += 1
    return pos_clusters[:num_pos_clusters], neg_clusters[:num_neg_clusters]


PositiveDetection = collections.namedtuple("PositiveDetection",
                                           ("cluster_start_end"
                                            +" cluster_max_peak_loc"
                                            +" cluster_max_peak_val"
                                            # these are the lengths of the
                                            # templates for where the
                                            # detection occurs
                                            +" cluster_detect_lengths"
                                            # these are the identities
                                            # of what the spiking detected
                                            # templates are
                                            +" cluster_detect_ids"
                                            # these are the observed 
                                            # detection values
                                            +" cluster_vals"
                                            +" true_label_times"
                                            +" phn_context"
                                            +" flt_context"
                                            +" utterances_path"
                                            +" file_index"))

FalsePositiveDetection = collections.namedtuple("FalsePositiveDetection",
                                           ("cluster_start_end"
                                            +" cluster_max_peak_loc"
                                            +" cluster_max_peak_val"
                                            +" cluster_max_peak_phn"
                                            # these are the lengths of the
                                            # templates for where the
                                            # detection occurs
                                            +" cluster_detect_lengths"
                                            # these are the identities
                                            # of what the spiking detected
                                            # templates are
                                            +" cluster_detect_ids"
                                            # these are the observed 
                                            # detection values
                                            +" cluster_vals"
                                            +" phn_context"
                                            +" flt_context"
                                            +" utterances_path"
                                            +" file_index"))


    
    

FalseNegativeDetection = collections.namedtuple("FalseNegativeDetection",
                                                ("true_window_cluster"
                                                 +" max_peak_loc"
                                                 +" max_peak_val"
                                                 +" window_vals"
                                                 +" true_label_times"
                                                 +" phn_context"
                                                 +" flt_context"
                                                 +" utterances_path"
                                                 +" file_index"))

def get_max_peak(cluster_window):
    """
    Parameters:
    ===========
    cluster_window:  np.ndarray
        We assume that the first and last values of the cluster window
        are not actually in the cluster, they are there just to assist
        with the peak finding

    Output:
    ======
    max_peak_loc: int
        Location of the maximum peak or -1 if none is found
    """
    potentials = np.arange(len(cluster_window)-2)[
        (cluster_window[1:-1] >= cluster_window[:-2])
        * (cluster_window[1:-1] >= cluster_window[2:])]
    potential_vals = cluster_window[1:-1][potentials]
    if len(potential_vals) == 0:
        return -1
    else:
        max_potential_id = np.argmax(potential_vals)
        return 1 + potentials[max_potential_id]





def get_pos_false_pos_false_neg_detect_points(detection_clusters_at_threshold,
                                              detection_array,
                                              detection_template_ids,
                                              template_lengths,
                                              window_start,
                                              window_end,example_start_end_times,
                                              utterances_path,
                                              file_indices,
                                              verbose=True,
                                              return_example_types=False):
    """
    Parameters:
    ===========
    detection_clusters_at_threshold:
    detection_array:
    C1:
    window_start:
    window_end:
    example_start_end_times:

    Output:
    =======
    pos_times - come with the point of maximal detection and the true point
    false_pos_times - just the point of maximal detection (obviously no true detection)
    false_neg_clusters - comes with the point of maximal detection and the true point
    """
    num_utts = detection_array.shape[0]
    pos_times = tuple([] for i in xrange(num_utts))
    false_pos_times = tuple([] for i in xrange(num_utts))
    false_neg_times = tuple([] for i in xrange(num_utts))
    if return_example_types:
        num_examples = sum(len(k) for k in example_start_end_times)
        # example types are 'p', 'fp', 'fn'
        example_types = np.zeros(num_examples,dtype=np.uint8)
        cur_example_for_typing=0
    for utt_id, utt_info in enumerate(itertools.izip(detection_clusters_at_threshold,detection_array,example_start_end_times)):
        phns = np.load(utterances_path+file_indices[utt_id]+'phns.npy')
        flts = np.load(utterances_path+file_indices[utt_id]+'feature_label_transitions.npy')
        
        if verbose:
            print utt_id
        detect_clusters, detection_row, start_end_times = utt_info
        # get the true positives and the false positives
        # mark which start end times have been detected
        examples_not_detected = np.ones(len(start_end_times),dtype=bool)
        for c in detect_clusters:
            is_false_pos = True
            for example_id,se in enumerate(start_end_times):
                s,e = se
                if s+window_start < c[1] and s+window_end >= c[0]:
                    is_false_pos = False
                    if return_example_types:
                        example_types[cur_example_for_typing+example_id] =1

                    #
                    # Confusing transformation here:
                    #   we need a value at the front and back of our
                    #   cluster in order to evaluate whether values
                    #   within the cluster count as local maximal
                    #   so we work with an extended cluster that includes
                    #   these head and tail points
                    #   but, once we find the largest local maximum
                    #   we then need to remove those points
                    cluster_vals = detection_row[c[0]-1:c[1]+1]
                    cluster_max_peak_loc = get_max_peak(cluster_vals)
                    if cluster_max_peak_loc == -1:
                        cluster_max_peak_val = -np.inf
                    else:
                        cluster_max_peak_val=cluster_vals[cluster_max_peak_loc]
                    # map the peak location and the
                    # cluster length back to the original system
                    # getting rid of the extended cluster
                    cluster_max_peak_loc -= 1
                    cluster_vals = cluster_vals[1:-1]
                    cluster_detect_lengths = np.array([template_lengths[idx] for idx in detection_template_ids[utt_id,c[0]:c[1]]])
                    cluster_detect_ids = detection_template_ids[utt_id,c[0]:c[1]]
                    
                    phn_context,flt_context = gtrd.get_phn_context(c[0],
                                                              c[1],
                                                              phns,
                                                              flts,
                                                              offset=1,
                                                              return_flts_context=True)
                    pos_times[utt_id].append(PositiveDetection(
                            cluster_start_end=c,
                            cluster_max_peak_loc = cluster_max_peak_loc,
                            cluster_max_peak_val=cluster_max_peak_val, 
                            cluster_detect_lengths=cluster_detect_lengths,
                            cluster_detect_ids=cluster_detect_ids,
                            cluster_vals=cluster_vals,
                            true_label_times=se,
                            phn_context=phn_context,
                            flt_context=flt_context,
                            utterances_path=utterances_path,
                            file_index=file_indices[utt_id]))
                    examples_not_detected[example_id] = False
            if is_false_pos:

                if c[0]-c[1] < 2:
                    cluster_max_peak_loc=0
                    cluster_max_peak_val=detection_row[c[0]]
                    cluster_vals = detection_row[c[0]:c[1]]
                else:
                    cluster_vals = detection_row[c[0]-1:c[1]+1]
                    cluster_max_peak_loc = get_max_peak(cluster_vals)
                    if cluster_max_peak_loc == -1:
                        cluster_max_peak_val = -np.inf
                    else:
                        cluster_max_peak_val=cluster_vals[cluster_max_peak_loc]
                        #
                        # Confusing transformation here:
                        #   we need a value at the front and back of our
                        #   cluster in order to evaluate whether values
                        #   within the cluster count as local maximal
                        #   so we work with an extended cluster that includes
                        #   these head and tail points
                #   but, once we find the largest local maximum
                #   we then need to remove those points
                        
                        # We map the peak location and the
                # cluster length back to the original system
                # getting rid of the extended cluster
                    cluster_max_peak_loc -= 1
                    cluster_vals = cluster_vals[1:-1]
                cluster_detect_lengths = np.array([template_lengths[idx] for idx in detection_template_ids[utt_id,c[0]:c[1]]])
                cluster_detect_ids = detection_template_ids[utt_id,c[0]:c[1]]

                phn_context,flt_context = gtrd.get_phn_context(c[0],
                                                          c[1],
                                                          phns,
                                                          flts,
                                                          offset=1,
                                                          return_flts_context=True)
                
                false_pos_times[utt_id].append(FalsePositiveDetection(
                        cluster_start_end=c,
                        cluster_max_peak_loc = cluster_max_peak_loc,
                        cluster_max_peak_val=cluster_max_peak_val, 
                        cluster_max_peak_phn=gtrd.get_phn_context(
                            cluster_max_peak_loc,
                            cluster_max_peak_loc+1,
                            phns,
                            flts,
                            offset=0,
                            return_flts_context=False),
                        
                        cluster_detect_lengths=cluster_detect_lengths,
                        cluster_detect_ids=cluster_detect_ids,
                        cluster_vals=cluster_vals,
                        phn_context=phn_context,
                        flt_context=flt_context,
                        utterances_path=utterances_path,
                        file_index=file_indices[utt_id]))
 
        for idx in np.arange(len(start_end_times))[examples_not_detected]:

            s = start_end_times[idx][0]
            e = start_end_times[idx][1]
            if verbose:
                print "False Negative in utterance %d at frame %d" % (utt_id,
                                                                      s)
            c = (s+window_start,s+window_end)
            cluster_vals = detection_row[c[0]-1:c[1]+1]
            cluster_max_peak_loc = get_max_peak(cluster_vals)
            if cluster_max_peak_loc == -1:
                cluster_max_peak_val = -np.inf
            else:
                cluster_max_peak_val=cluster_vals[cluster_max_peak_loc]
            #
            # Confusing transformation here:
            #   we need a value at the front and back of our
            #   cluster in order to evaluate whether values
            #   within the cluster count as local maximal
            #   so we work with an extended cluster that includes
            #   these head and tail points
            #   but, once we find the largest local maximum
            #   we then need to remove those points
                
            # We map the peak location and the
            # cluster length back to the original system
            # getting rid of the extended cluster
            cluster_max_peak_loc -= 1

            cluster_vals = cluster_vals[1:-1]
            cluster_detect_lengths = np.array([template_lengths[idx] for idx in detection_template_ids[utt_id,c[0]:c[1]]])
            cluster_detect_ids = detection_template_ids[utt_id,c[0]:c[1]]
            phn_context,flt_context = gtrd.get_phn_context(c[0],
                                                      c[1],
                                                      phns,
                                                      flts,
                                                      offset=1,
                                                      return_flts_context=True)

            false_neg_times[utt_id].append(FalseNegativeDetection(
                    true_window_cluster=c,
                    max_peak_loc=cluster_max_peak_loc,
                    max_peak_val=cluster_max_peak_val,
                    window_vals=cluster_vals,
                    true_label_times=(s,e),
                    phn_context=phn_context,
                    flt_context=flt_context,
                    utterances_path=utterances_path,
                    file_index=file_indices[utt_id]))

        if return_example_types:
            cur_example_for_typing+=len(start_end_times)
            assert cur_example_for_typing <= num_examples
    if return_example_types:
        return pos_times, false_pos_times, false_neg_times, example_types
    else:
        return pos_times, false_pos_times, false_neg_times


def get_false_positives(false_pos_times,S_config,E_config,
                       offset=0,
                        waveform_offset=0,
                        verbose=False):
    return_false_positives = []
    for utt_id, utt_false_positives in enumerate(false_pos_times):
        if len(utt_false_positives)== 0: 
            return_false_positives.append([])
            continue
        # we know its non-empty
        # will open the data
        print "utt_id=%d" %utt_id
        # for fp_id, fp in enumerate(utt_false_positives):
        #     print (fp.cluster_start_end[0]+fp.cluster_max_peak_loc
        #              + fp.cluster_detect_lengths[fp.cluster_max_peak_loc] - 
        #            fp.cluster_start_end[0]+fp.cluster_max_peak_loc)
        #    print "fp_id=%d" %fp_id
        return_false_positives.append( gtrd.get_syllable_features_cluster(
                utt_false_positives[0].utterances_path,
                utt_false_positives[0].file_index,
                tuple(
                    (fp0.cluster_start_end[0]+fp0.cluster_max_peak_loc,
                     fp0.cluster_start_end[0]+fp0.cluster_max_peak_loc
                     + fp0.cluster_detect_lengths[fp0.cluster_max_peak_loc])
                    for fp0 in utt_false_positives),
                S_config=S_config,
                E_config=E_config,
                offset = offset,
                E_verbose=False,
                # we aren't estimating background here at all
                avg_bgd=None,
                waveform_offset=waveform_offset,
                assigned_phns = (utt_false_positives[0].cluster_max_peak_phn,)))
    return tuple(return_false_positives)

def get_true_positives(true_pos_times,S_config,E_config,
                       offset=0,
                        waveform_offset=0,
                        verbose=False):
    return_true_positives = []
    for utt_id, utt_true_positives in enumerate(true_pos_times):
        if len(utt_true_positives)== 0: 
            return_true_positives.append([])
            continue
        # we know its non-empty
        # will open the data
        print "utt_id=%d" %utt_id
        # for fp_id, fp in enumerate(utt_true_positives):
        #     print (fp.cluster_start_end[0]+fp.cluster_max_peak_loc
        #              + fp.cluster_detect_lengths[fp.cluster_max_peak_loc] - 
        #            fp.cluster_start_end[0]+fp.cluster_max_peak_loc)
        #    print "fp_id=%d" %fp_id
        return_true_positives.append( gtrd.get_syllable_features_cluster(
                utt_true_positives[0].utterances_path,
                utt_true_positives[0].file_index,
                tuple(
                    (fp0.cluster_start_end[0]+fp0.cluster_max_peak_loc,
                     fp0.cluster_start_end[0]+fp0.cluster_max_peak_loc
                     + fp0.cluster_detect_lengths[fp0.cluster_max_peak_loc])
                    for fp0 in utt_true_positives),
                S_config=S_config,
                E_config=E_config,
                offset = offset,
                E_verbose=False,
                # we aren't estimating background here at all
                avg_bgd=None,
                waveform_offset=waveform_offset))
    return tuple(return_true_positives)

def get_false_negatives(false_negative_times,S_config,E_config,
                       offset=0,
                        waveform_offset=0,
                        verbose=False):
    return_false_negatives = []
    for utt_id, utt_false_negatives in enumerate(false_negative_times):
        if len(utt_false_negatives)== 0: 
            return_false_negatives.append([])
            continue
        # we know its non-empty
        # will open the data
        print "utt_id=%d" %utt_id
        # for fp_id, fp in enumerate(utt_false_negatives):
        #     print (fp.cluster_start_end[0]+fp.cluster_max_peak_loc
        #              + fp.cluster_detect_lengths[fp.cluster_max_peak_loc] - 
        #            fp.cluster_start_end[0]+fp.cluster_max_peak_loc)
        #    print "fp_id=%d" %fp_id
        return_false_negatives.append( gtrd.get_syllable_features_cluster(
                utt_false_negatives[0].utterances_path,
                utt_false_negatives[0].file_index,
                tuple(
                    fp0.true_label_times
                    for fp0 in utt_false_negatives),
                S_config=S_config,
                E_config=E_config,
                offset = offset,
                E_verbose=False,
                # we aren't estimating background here at all
                avg_bgd=None,
                waveform_offset=waveform_offset))
    return tuple(return_false_negatives)


def get_false_pos_clusters(Es_false_pos,
                              templates,
                              template_ids):
    return tuple(
        Es_false_pos[template_ids==i][:,:templates[i].shape[0]]
        for i in xrange(len(templates)))



def recover_template_ids_detect_times(detect_times):
    return_template_ids = np.array([],dtype=int)
    for utt_id, utt_detect_times in enumerate(detect_times):
        if len(utt_detect_times) == 0: continue

        return_template_ids = np.append(
            return_template_ids, 
            tuple(
                fp.cluster_detect_ids[fp.cluster_max_peak_loc]
                for fp in utt_detect_times))

    return return_template_ids
            
       



def map_cluster_responses_to_grid(cluster_responses):
    cluster_length = cluster_responses.shape[1]
    response_grid = np.zeros((cluster_length,cluster_length))
    response_points = np.arange(cluster_length) * (cluster_responses.max() - cluster_responses.min())/cluster_length + cluster_responses.min()
    for col_idx,response_col in enumerate(cluster_responses.T):
        col_pdf = gaussian_kde(response_col)
        response_grid[col_idx] = col_pdf(response_points + np.random.randn(len(response_points))*np.std(response_points)/1000)
    return response_grid.T,response_points

def display_response_grid(fname,response_grid,response_points,point_spacing=10):
    plt.close()
    plt.imshow(response_grid[::-1])
    plt.yticks(np.arange(response_points.shape[0])[::10],response_points[::-point_spacing].astype(int))
    plt.savefig(fname)
    
