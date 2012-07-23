import numpy as np
import numpy.testing
import fast_like
root_path = '/home/mark/projects/Template-Speech-Recognition/'

import sys, os, cPickle
sys.path.append(root_path)
import template_speech_rec.test_template as tt


# generate random template

T_num_rows = 4
T_num_cols = 6

T = np.maximum(.05,
                np.minimum(.95,
                            np.random.rand(4*6).astype(np.float64).reshape(T_num_rows,
                                                                         T_num_cols)))

#
# Test what happens when
#     E_window is longer
#     E_window is shorter
#     E_window is the same size

# number of time points is the number of columns
E_num_cols = T_num_cols
E_window = np.random.randint(2,size=(T_num_rows*
                                     E_num_cols)).reshape(T_num_rows,E_num_cols).astype(np.uint8)
num_detections = 1
pad_front = 0
pad_back = 0


W_bg = np.empty(T_num_cols,dtype=np.float64)
detect_sums = np.zeros(num_detections, dtype=np.float64)
bg = np.maximum(.05,
                 np.random.rand(T_num_rows).astype(np.float64)*.4)

C_inv_long = np.empty((T_num_rows,T_num_cols),dtype=np.float64)
W = np.empty((T_num_rows,T_num_cols),dtype=np.float64)

C = fast_like.fast_like(T,
                         bg,
                         C_inv_long,
                         W,
                         T_num_rows,
                         T_num_cols,
                         E_window,
                         E_num_cols,
                         num_detections,
                             pad_front,
                             W_bg,
                             detect_sums)





template_length = T.shape[1]
U_bgd = np.tile(bg,(template_length,1)).transpose()
T_inv = 1. - T
U_bgd_inv = 1. - U_bgd
C_exp_inv_long2 = np.log(T_inv/U_bgd_inv)
# get the C_k
C2 = (C_exp_inv_long2).sum()
W2 = np.log((T/U_bgd)) - C_exp_inv_long2

W_bg2 = np.dot(bg,W2)

np.testing.assert_almost_equal(C,C2)
np.testing.assert_array_almost_equal(C_exp_inv_long2,C_inv_long)
np.testing.assert_array_almost_equal(W2,W)
np.testing.assert_array_almost_equal(W_bg,W_bg2)


#
# now we are going to test what
# happens when we add padding, right now
# the length of the template and the length of th ewindow
# are the same so there should never be back padding
# we are just going to see the effects of front padding


pad_front =2

detect_sums[:] = 0

C = fast_like.fast_like(T,
                         bg,
                         C_inv_long,
                         W,
                         T_num_rows,
                         T_num_cols,
                         E_window,
                         E_num_cols,
                         num_detections,
                             pad_front,
                             W_bg,
                             detect_sums)

np.testing.assert_almost_equal(detect_sums[0],np.sum(W_bg[:2]))

pad_front = 0
num_detections = 4
del detect_sums
detect_sums = np.zeros(num_detections, dtype=np.float64)

C = fast_like.fast_like(T,
                         bg,
                         C_inv_long,
                         W,
                         T_num_rows,
                         T_num_cols,
                         E_window,
                         E_num_cols,
                         num_detections,
                             pad_front,
                             W_bg,
                             detect_sums)

np.testing.assert_almost_equal(
    detect_sums,
    np.array([0,0,W_bg[-1],np.sum(W_bg[-2:])]))


#
# Now we are testing whether convolution is behaving like it is supposed to
#
#

E_row = np.array([0,1,0,1],dtype=np.uint8)
W_row = np.array([.6,.8,.9,1.4],dtype=np.float64)

E_num_cols = 4
W_num_cols = 4
num_detections = 4
detect_sums = np.zeros(num_detections,dtype=np.float64)
pad_front = 0

fast_like.test_convolution(E_row,
                           W_row,
                           detect_sums,
                           E_num_cols,
                           W_num_cols,
                            num_detections,
                           pad_front)


def slow_convolution(E_row,
                     W_row,
                     detect_sums,
                     E_num_cols,
                     W_num_cols,
                     num_detections,
                     pad_front):
    start_detect_idx = max(0, pad_front- W_num_cols+1)
    print "start_detect_idx",start_detect_idx
    # check to make sure that we have some detections to do
    # position in out_convolve that we begin in
    start_out_idx = max(0,W_num_cols-1-pad_front)
    print "start_out_idx", start_out_idx
    num_out_idx = W_num_cols-1 + E_num_cols
    print "num_out_idx", num_out_idx 
    # the starting detect idx is always 0 
    # want to know the end E idx, 
    # couple of situations
    # in the case where start_out_idx = W_num_cols-1 or pad_front = 0
    # then num_detections- start_detect_idx is the last position 
    # so the last E index to use is int_min( E_num_cols,W_num_cols-1 + num_detections - start_detect_idx)
    # and then in the case where pad_front = 1
    # int_min( E_num_cols,W_num_cols-1 + num_detections - start_detect_idx-1)
    # then we get
    # int_min( E_num_cols,W_num_cols-1 + num_detections - start_detect_idx-pad_front)
    end_E_idx = min(E_num_cols-1,
                                    W_num_cols-1 
                    + num_detections-1
                    - start_detect_idx 
                    - pad_front)
    print "end_E_idx", end_E_idx
    for E_idx in range(0,end_E_idx+1):
        print E_row[E_idx]
        if E_row[E_idx] > 0:
            # need to figure out all the detect indices that this guy
            # will contribute to
            # if E_idx = 0 will definitely contribute to out_convolve[E_idx]
            # out_convolve[E_idx+1], ..., out_convolve[E_idx+W_num_cols-1]
            # end_out_idx = int_min(E_idx+W_num_cols-1, 
            #                      num_out_idx)
            # which means that the detect_indices are in th erange
            # (E_idx+start_detect_idx,
            #     end_out_idx - start_out_idx + start_detect_idx)
            # which entries of W are we then going to be ranging over?
            # in the case where E_idx = 0 and pad_front = 0
            # we only are multiplied by W[0]
            # if E_idx = 0 and pad_front = 1
            # we will get multiplied by W[0] and W[1]
            # so for E_idx = 0 we range over W[0] to W[k]
            # where k = int_min( W_num_cols-1, pad_front)
            # if E_idx = 1 and pad_front = 0 and num_detections >= 1
            #    we range over W[0] and W[1]
            # if E_idx = 1 and pad_front = 1 and num_detections >= 2
            # we get W[0], W[1], W[2]
            #
            # the basic idea is that we have a w_max coordinate
            # w_max = int_min(pad_front, W_num_cols - 1-E_idx) + E_idx
            # w_min = int_max(0,w_max - num_detections )
            # num_detections determines our w_min
            w_max_idx = min(pad_front  + E_idx,
                            W_num_cols-1)
            w_min_idx = max(0,w_max_idx - num_detections+1)
            detect_sums_min_idx = start_detect_idx 
            print "detect_sums_min_idx", detect_sums_min_idx
            max_detect_idx = min(E_idx+pad_front,
                               num_detections-1)
            min_detect_idx = max(0,E_idx-W_num_cols+1
                                 +pad_front)
            print "max_detect_idx", max_detect_idx
            print "min_detect_idx", min_detect_idx
            for detect_idx in range(min_detect_idx,
                                    max_detect_idx+1):
                print "W_idx", E_idx-detect_idx
                print "detect sums idx", detect_idx
                detect_sums[detect_idx] += W_row[E_idx-detect_idx+pad_front]


detect_sums[:] =0            
num_detections = 2
slow_convolution(E_row,
                     W_row,
                     detect_sums,
                     E_num_cols,
                     W_num_cols,
                     num_detections,
                     pad_front)

detect_sums[:] =0            
num_detections = 4
pad_front = 1
slow_convolution(E_row,
                     W_row,
                     detect_sums,
                     E_num_cols,
                     W_num_cols,
                     num_detections,
                     pad_front)

detect_sums[:] =0            
num_detections = 4
pad_front = 2
slow_convolution(E_row,
                     W_row,
                     detect_sums,
                     E_num_cols,
                     W_num_cols,
                     num_detections,
                     pad_front)


def slow_simple_convolution(E_row,
                     W_row,
                     detect_sums,
                     E_num_cols,
                     W_num_cols,
                     num_detections,
                     pad_front):
    if pad_front > 0:
        E_row = np.hstack((np.zeros(pad_front,
                                         np.uint8),
                                E_row))
        E_num_cols += pad_front
    return _slow_simple_convolution(E_row,
                                    W_row,
                                    detect_sums,
                                    E_num_cols,
                                    W_num_cols,
                                    num_detections)

def _slow_simple_convolution(E_row,
                                    W_row,
                                    detect_sums,
                                    E_num_cols,
                                    W_num_cols,
                                    num_detections):
    for d in xrange(num_detections):
        detect_sums[d] = np.dot(E_row[d:min(W_num_cols+d,
                                     E_num_cols)],
                                W_row[:min(E_num_cols-d,
                                           W_num_cols)])



slow_simple_convolution(E_row,
                     W_row,
                     detect_sums,
                     E_num_cols,
                     W_num_cols,
                     num_detections,
                     pad_front)


#
# The final show-down
# a comparison between my cython version and the
# one I wrote a while back
#


T_num_rows = 4
T_num_cols = 6

T = np.maximum(.05,
                np.minimum(.95,
                            np.random.rand(4*6).astype(np.float64).reshape(T_num_rows,
                                                                         T_num_cols)))

#
# Test what happens when
#     E_window is longer
#     E_window is shorter
#     E_window is the same size

# number of time points is the number of columns
E_num_cols = T_num_cols
E_window = np.random.randint(2,size=(T_num_rows*
                                     E_num_cols)).reshape(T_num_rows,E_num_cols).astype(np.uint8)
num_detections = 1
pad_front = 0
pad_back = 0


W_bg = np.empty(T_num_cols,dtype=np.float64)
detect_sums = np.zeros(num_detections, dtype=np.float64)
bg = np.maximum(.05,
                 np.random.rand(T_num_rows).astype(np.float64)*.4)

C_inv_long = np.empty((T_num_rows,T_num_cols),dtype=np.float64)
W = np.empty((T_num_rows,T_num_cols),dtype=np.float64)


s,C = tt.score_template_background_section(T,bg,E_window,front_bgd_pad=0,back_bgd_pad=0)

C2 = fast_like.fast_like(T,
                         bg,
                         C_inv_long,
                         W,
                         T_num_rows,
                         T_num_cols,
                         E_window,
                         E_num_cols,
                         num_detections,
                             pad_front,
                             W_bg,
                             detect_sums)

detect_sums[:] = 0.
E_windows = E_window.reshape(1,4,6)
E_num_cols_array = np.array([E_num_cols,
                             E_num_cols])
num_E_windows =1
T_flat = T.ravel()
bgs = bg.reshape(1,bg.shape[0])

C3s = fast_like.array_fast_like(T,
                     bgs,
                     T_num_rows,
                     T_num_cols,
                     E_windows,
                     E_num_cols_array,
                     num_detections,
                     pad_front)


E_window2 = np.random.randint(2,size=(T_num_rows*
                                     E_num_cols)).reshape(T_num_rows,E_num_cols).astype(np.uint8)

bg2 = np.maximum(.05,
                 np.random.rand(T_num_rows).astype(np.float64)*.4)

C4 = fast_like.fast_like(T,
                         bg2,
                         C_inv_long,
                         W,
                         T_num_rows,
                         T_num_cols,
                         E_window2,
                         E_num_cols,
                         num_detections,
                             pad_front,
                             W_bg,
                             detect_sums)

E_windows = np.array([E_window, E_window2])
T2 = np.maximum(.05,
                np.minimum(.95,
                            np.random.rand(4*6).astype(np.float64).reshape(T_num_rows,
                                                                         T_num_cols)))

Ts = np.array([T,T2])

bgs = np.array([bg, bg2])

E_num_cols_array = np.array([E_window.shape[1],
                             E_window2.shape[1]])
Cs,detect_sums = fast_like.templates_examples_fast_like(Ts,bgs,E_windows,
                    E_num_cols_array,
                    num_detections,
                    pad_front)

C4,detect_sums = fast_like.array_fast_like(T,
                         np.array([bg]),
                         T_num_rows,
                         T_num_cols,
                         np.array([E_window]),
                         np.array([E_num_cols]),
                         num_detections,
                             pad_front)
