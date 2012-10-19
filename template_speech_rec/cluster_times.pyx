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