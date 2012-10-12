import numpy as np
A = np.array(
    [
        [[1,0,0,0],
         [0,1,1,0],
         [0,0,1,1],
         [1,0,0,1]],
        [[0,0,1,0],
         [1,1,1,0],
         [0,0,0,1],
         [0,1,0,0]]]).astype(np.uint8)

code_blocks = np.array(
    [
        [
            [[0,0],
             [0,1]],
            [[1,1],
             [0,0]]],
        [
            [[1,0],
             [0,1]],
            [[0,0],
             [1,1]]],
        [
            [[0,0],
             [1,0]],
            [[1,0],
             [1,0]]]]).astype(np.uint8)

        
from scipy import ndimage

ndimage.generic_filter(np.arange(10.),np.mean,size=3,mode='constant')

B = ndimage.correlate(A,code_blocks[1],mode='constant')
# problem is to figure out the submatrix
# in this case the detection submatrix is
B[1:,1:,1:]

d = np.array([1,0]).reshape(2,1,1)

C = ndimage.correlate(A,d,mode='constant')

# conjecture:
# if the weights are in an array w
# w.shape = (d0,d1,d2)
# then the sub array is Out[d0-1:,d1-1:,d2-1:]

#
# TODO for next time
# have the features be output in a 3D matrix
# change edge signal processing so that occurs
# use the feature estimation code to estimate the waliji parts or use
# the estimated waliji parts correctly
# the simple function is for determining parts is going to be

E = np.arange(24).reshape(2,3,4)

def reorg_E(E):
    """
    E is assumed to have features in the zeroth axis and
    time along the 1st axis, the features are in groups according to frequency
    We want time along the zero axis and we want frequency along the 1st axis
    and edge type along the second axis
    """
    feature_height = E.shape[0]/8
    return np.array([E[i*feature_height:(i+1)*feature_height].T
                     for i in xrange(8)]).swapaxes(0,1).swapaxes(1,2)

def code_parts(E,part_blocks,E_is_2d=True):
    """
    Assumes that E has been reorganized to have
    dimension 0 be the time axis
    dimension 1 be the frequency axis
    dimension 2 be the edge type axis
    
    """
    part_block_shape = part_blocks[0].shape
    return np.argmax(np.array([
            ndimage.correlate(E,part_block)[part_block_shape[0]-1:,
                                            part_block_shape[1]-1:]
            for part_block in part_blocks]),0)


    
    
ndimage.correlate(np.ones((5,3,4)),np.arange(36).reshape(3,3,4),
                  mode='constant')

ndimage.correlate(np.ones((5,4,4)),np.arange(36).reshape(3,3,4),
                  mode='constant')

ndimage.correlate(np.ones((5,4,5)),np.arange(36).reshape(3,3,4),
                  mode='constant')

ndimage.correlate(np.ones((5,4,6)),np.arange(36).reshape(3,3,4),
                  mode='constant')

ndimage.correlate(np.ones((5,4,4)),np.arange(48).reshape(4,3,4),
                  mode='constant')

np.argmax(ndimage.correlate(np.ones((6,5,8)),np.arange(6*5*8).reshape(6,5,8),
                  mode='constant'))

np.max(ndimage.correlate(np.ones((6,5,8)),np.arange(6*5*8).reshape(6,5,8),
                  mode='constant'))


np.max(ndimage.correlate(np.ones((2,5,7)),np.arange(2*5*7).reshape(2,5,7),
                  mode='constant'))
