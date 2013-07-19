# cython cluster_times.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#      -I/usr/include/python2.7 -o cluster_times.so cluster_times.c
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float32
UINT = np.uint8
ctypedef np.float32_t DTYPE_t

ctypedef np.uint8_t UINT_t

cdef int longest_cluster_detect_id(np.ndarray[ndim=1,dtype=int] times,
                                               np.ndarray[ndim=1,dtype=int] detection_lengths,
                                               np.ndarray[ndim=1,dtype=np.uint8_t] cluster_partition,
                                               int num_detections):
    """
    Gets the index for the longest cluster within the lot
    """
    cdef int longest_cluster_id = -1
    cdef int cluster_length = 0
    cdef int k, longest_cluster_length, new_cluster_length, cur_cluster_start_id
    # if the longest cluster less the detection length is zero then we don't
    # go beyond this point
    longest_cluster_length = 0
    cur_cluster_start_id = 0
    for k in range(1,num_detections):
        # detect when we have a split at k-1
        if cluster_partition[k] > 0:
            new_cluster_length = times[k-1] - times[cur_cluster_start_id] - detection_lengths[cur_cluster_start_id]
            if longest_cluster_length < new_cluster_length:
                # print "new_cluster_length = %d, longest_cluster_length = %d" % (new_cluster_length, longest_cluster_length)
                # print "k=%d\tcur_cluster_start_id=%d" % (k,cur_cluster_start_id)

                longest_cluster_id = cur_cluster_start_id
                longest_cluster_length = new_cluster_length
                
            # update the fact that the cluster now starts at $k$
            cur_cluster_start_id = k
    return longest_cluster_id
                
cdef void split_cluster(np.ndarray[ndim=1,dtype=int] times,
                        np.ndarray[ndim=1,dtype=np.uint8_t] cluster_partition,
                        int num_detections,
                        int longest_cluster_id):
    """
    Basic assumption is that the cluster has more than one point in it
    This assumption is ensured by the proper performance of longest_cluster_detect_id
    """
    cdef int split_idx = 1 + longest_cluster_id
    cdef int cur_jump
    cdef int max_jump = 0
    cdef int max_jump_loc = 0
    while cluster_partition[split_idx] < 1 and split_idx < num_detections:
        # loop invariance condition ensures we stay within the cluster and that we don't
        # exceed the number of detections
        cur_jump = times[split_idx] -times[split_idx-1]
        if cur_jump > max_jump:
            max_jump_loc = split_idx
            max_jump = cur_jump

        split_idx += 1

    cluster_partition[split_idx] = 1
            
    

def cluster_times_variable_length(
    np.ndarray[ndim=1,dtype=int] times,
    np.ndarray[ndim=1,dtype=int] detection_lengths):
    """
    times:
        just the times where I found detections
    detection_lengths:
        the lengths for each of those detections found
    num_times
        number of scores computed over the utterance
    """
    cdef int num_detections = times.shape[0]
    cdef np.ndarray[ndim=1,dtype=np.uint8_t] cluster_partition = np.zeros(num_detections,dtype=np.uint8)
    # codes for the cluster partition are
    # 1 means that this is where a cluster starts and further
    #    processing may be needed to get it to obey that the length
    #    of the cluster is less than C1
    # 2 means that this is where a cluster ends and no further processing is required


    # start off with one big cluster
    cluster_partition[0] = 1
    # check whether there is any more processing to do
    if num_detections < 2:
        return cluster_partition


    # now we divide the cluster up as much as possible
    cdef int time_idx = 0
    for time_idx in range(1,num_detections):
        if times[time_idx] - times[time_idx-1] >= detection_lengths[time_idx-1]:
            cluster_partition[time_idx] = 1

    
    cdef int longest_cluster_id

    longest_cluster_id = longest_cluster_detect_id(times,
                                               detection_lengths,
                                               cluster_partition,
                                               num_detections)

    cdef int num_iter = 0
    # print "longest_cluster_id = %d" % longest_cluster_id
    # print cluster_partition
    while longest_cluster_id > -1 and num_iter < times[num_detections-1]:
        # print "in loop"
        split_cluster(times,
                      cluster_partition,
                      num_detections,
                      longest_cluster_id)
        
        longest_cluster_id = longest_cluster_detect_id(times,
                                               detection_lengths,
                                               cluster_partition,
                                               num_detections)
        num_iter += 1
        # print "longest_cluster_id = %d" % longest_cluster_id
    return cluster_partition
    

def cluster_times(np.ndarray[ndim=1,dtype=np.uint16_t] times,
                  DTYPE_t C0,
                  DTYPE_t C1):
    """
    Assumed to take in maximal points
    """
    cdef np.uint16_t num_times = times.shape[0]
    cdef np.ndarray[ndim=1,dtype=np.uint8_t] cluster_partition = np.zeros(num_times+1,dtype=np.uint8)
    # codes for the cluster partition are
    # 1 means that this is where a cluster starts and further
    #    processing may be needed to get it to obey that the length
    #    of the cluster is less than C1
    # 2 means that this is where a cluster starts and no further processing is
    #  needed
    cluster_partition[0] = 1
    cluster_partition[num_times] = 2
    cdef np.uint16_t time_idx = 0
    for time_idx in range(1,num_times):
        if times[time_idx] - times[time_idx-1] >= C0:
            cluster_partition[time_idx] = 1

    cdef np.uint16_t cluster_start_id = 0
    cdef np.uint16_t max_dist = 0
    cdef np.uint16_t total_dist = 0
    cdef np.uint16_t cur_dist = 0
    cdef np.uint16_t max_link_id
    while cluster_start_id < num_times:
        if cluster_partition[cluster_start_id+1] > 0:
            cluster_partition[cluster_start_id] = 2
            cluster_start_id += 1
            while (cluster_partition[cluster_start_id] != 1) and (cluster_start_id < num_times):
                cluster_start_id += 1
        else:
            # Now we know that cluster_partition[cluster_start_id+1] == 0
            # This means that this cluster has more than one element so
            # it has a total_length and a max_length as well as a max_id
            # these three quantities are computed incrementally until
            # we get to the end of the cluster
            max_dist = 0
            total_dist = 0
            time_idx = 1
            #
            # time_idx will get incremented to T such that
            # cluster_partition[cluster_start_id +T] > 0
            # which means that the elements times[cluster_start_id:cluster_start_id+T]
            # are all included in this particular cluster, so that means that there 
            # could be a break at [cluster_start_id+1,...,cluster_start_id+T-1]
            #
            # where the break being at cluster_start_id+i means that we have a break
            # between times[cluster_start_id+i-1] and times[cluster_start_id+i]
            # so in that case the max_link_id = cluster_start_id+i
            while cluster_partition[cluster_start_id+time_idx] == 0:
                cur_dist = times[cluster_start_id+time_idx] - times[cluster_start_id+time_idx-1]
                if cur_dist > max_dist:
                    max_link_id = cluster_start_id+time_idx
                    max_dist = cur_dist
                total_dist += cur_dist
                time_idx += 1
            # At this point we have the maximum length link and we know the total 
            # distance, if the total distance is less than C1 hen we are done
            if total_dist > C1:
                # we have to but a break at the point with the largest
                # link, we then have two situations
                # either that link is within C1 times of cluster_start_id
                # in which case this front cluster is done,
                # or the link is too far
                if times[cluster_start_id + time_idx-1] - times[max_link_id] <= C1:
                    cluster_partition[max_link_id] = 2
                else:
                    cluster_partition[max_link_id] = 1
                if times[max_link_id-1] - times[cluster_start_id] <= C1:
                        cluster_partition[cluster_start_id] = 2
                        if cluster_partition[max_link_id] == 1:
                            cluster_start_id=max_link_id
                else:
                    cluster_partition[cluster_start_id] = 1
            else:
                cluster_partition[cluster_start_id] = 2
                #
                # finish off by finding the next cluster_start_id
                # we only get to this point if both cluster_start_id and max_link_id
                # have complete clusters
                cluster_start_id += time_idx
                while (cluster_partition[cluster_start_id] != 1) and (cluster_start_id < num_times):
                    cluster_start_id+= 1

    return cluster_partition

def cluster_times_template_lengths(np.ndarray[ndim=1,dtype=np.uint16_t] times,
                                   np.ndarray[ndim=1,dtype=np.uint16_t] template_lengths,
                                   np.ndarray[ndim=1,dtype=np.uint16_t] template_components):
    """
    Assumed to take in maximal points
    template_lengths helps us compute when two detections should be thrown
    into the same cluster
    template_components is the identities for the different detections
    """
    cdef np.uint16_t num_times = times.shape[0]
    cdef np.ndarray[ndim=1,dtype=np.uint8_t] cluster_partition = np.zeros(num_times+1,dtype=np.uint8)
    # codes for the cluster partition are
    # 1 means that this is where a cluster starts and further
    #    processing may be needed to get it to obey that the length
    #    of the cluster is less than C1
    # 2 means that this is where a cluster starts and no further processing is
    #  needed
    cluster_partition[0] = 1
    cluster_partition[num_times] = 2
    cdef np.uint16_t time_idx = 0
    for time_idx in range(1,num_times):
        if times[time_idx] - times[time_idx-1] >= template_lengths[template_components[time_idx-1]]:
            cluster_partition[time_idx] = 1

    cdef np.uint16_t cluster_start_id = 0
    cdef np.uint16_t max_dist = 0
    cdef np.uint16_t total_dist = 0
    cdef np.uint16_t cur_dist = 0
    cdef np.uint16_t max_link_id
    while cluster_start_id < num_times:
        if cluster_partition[cluster_start_id+1] > 0:
            cluster_partition[cluster_start_id] = 2
            cluster_start_id += 1
            while (cluster_partition[cluster_start_id] != 1) and (cluster_start_id < num_times):
                cluster_start_id += 1
        else:
            # Now we know that cluster_partition[cluster_start_id+1] == 0
            # This means that this cluster has more than one element so
            # it has a total_length and a max_length as well as a max_id
            # these three quantities are computed incrementally until
            # we get to the end of the cluster
            max_dist = 0
            total_dist = 0
            time_idx = 1
            #
            # time_idx will get incremented to T such that
            # cluster_partition[cluster_start_id +T] > 0
            # which means that the elements times[cluster_start_id:cluster_start_id+T]
            # are all included in this particular cluster, so that means that there 
            # could be a break at [cluster_start_id+1,...,cluster_start_id+T-1]
            #
            # where the break being at cluster_start_id+i means that we have a break
            # between times[cluster_start_id+i-1] and times[cluster_start_id+i]
            # so in that case the max_link_id = cluster_start_id+i
            while cluster_partition[cluster_start_id+time_idx] == 0:
                cur_dist = times[cluster_start_id+time_idx] - times[cluster_start_id+time_idx-1]
                if cur_dist > max_dist:
                    max_link_id = cluster_start_id+time_idx
                    max_dist = cur_dist
                total_dist += cur_dist
                time_idx += 1
            # At this point we have the maximum length link and we know the total 
            # distance, if the total distance is less than C1 hen we are done
            if total_dist > template_lengths[template_components[cluster_start_id]] * 1.5:
                # we have to but a break at the point with the largest
                # link, we then have two situations
                # either that link is within C1 times of cluster_start_id
                # in which case this front cluster is done,
                # or the link is too far
                if times[cluster_start_id + time_idx-1] - times[max_link_id] <= template_lengths[template_components[max_link_id]] * 1.5:
                    cluster_partition[max_link_id] = 2
                else:
                    cluster_partition[max_link_id] = 1
                if times[max_link_id-1] - times[cluster_start_id] <= template_lengths[template_components[cluster_start_id]] * 1.5:
                        cluster_partition[cluster_start_id] = 2
                        if cluster_partition[max_link_id] == 1:
                            cluster_start_id=max_link_id
                else:
                    cluster_partition[cluster_start_id] = 1
            else:
                cluster_partition[cluster_start_id] = 2
                #
                # finish off by finding the next cluster_start_id
                # we only get to this point if both cluster_start_id and max_link_id
                # have complete clusters
                cluster_start_id += time_idx
                while (cluster_partition[cluster_start_id] != 1) and (cluster_start_id < num_times):
                    cluster_start_id+= 1

    return cluster_partition
