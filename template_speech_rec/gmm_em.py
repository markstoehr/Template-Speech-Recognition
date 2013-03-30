#!/usr/bin/python

import numpy as np
import argparse, itertools



def get_data_vec(data_dir,save_tag):
    """
    Retrieves the spectral data examples and 
    puts them into a vector for clustering.
    Also constructs an array to record data about each frame.

    This is so that they can be passed to the HMM
    column one is the utterance id, column two is the number
    of frames in that utterance, column three is the frame number 
    within the utterance example that this frame is taken from
    (so it gets the time). With these data we can then pass
    the clustered MFCCs to the HMM

    
    """
    outfile = np.load('%sSs_lengths_%s.npz' % (data_dir,save_tag))
    Slengths = outfile['Slengths']
    Ss = outfile['Ss']
    numx = Slengths.sum()
    X = np.zeros((numx,Ss[0].shape[1]))

    X_metadata = np.zeros((numx,3))
    
    cur_X_idx = 0
    cur_frame_idx = 0
    cur_S_idx = 0
    while cur_X_idx < numx:
        X[cur_X_idx][:] = Ss[cur_S_idx][cur_frame_idx][:]
        X_metadata[cur_X_idx][0] = cur_S_idx
        X_metadata[cur_X_idx][1] = Slengths[cur_S_idx]
        X_metadata[cur_X_idx][2] = cur_frame_idx

        cur_frame_idx+= 1
        if cur_frame_idx == Slengths[cur_S_idx]:
            cur_frame_idx = 0
            cur_S_idx+= 1
        cur_X_idx+= 1
        
    return X, X_metadata

    

def kmeanspp_gmm_init(
    k,
    X,centers=None):
    """
    Construct a set of k centers to initialize kmeans
    or GMM clustering over the data

    Copied from David Arthur and Sergei Vassilvitskii


    Parameters:
    ===========
    k:  int
        Number of cluster centers to initialize
    X: numpy.ndarray[ndim=2]
        Data matrix, array of vectors, dimension 0 indexes
        along the data points and dimension 1 indexes along the coordinates
        of the data so X[i,j] is the $j$th coordinate of the $i$th 
        datum.
    """
    # first whiten X to make distance more meaningful
    numx = X.shape[0]
    xshape = X.shape[1:]
    xdim = np.prod(xshape)
    X = X.reshape(numx,xdim)
    
    m = np.mean(X,axis=0)
    X -= m
    s = np.std(X,axis=0)
    X /= s

    numx, xdim = X.shape

    if centers is None:
        centers = np.zeros((k,xdim))
        start_idx = 0
    else:
        orig_k = centers.shape[0]
        centers = centers.reshape(orig_k,xdim)
        k_prime = k - orig_k
        centers = np.vstack((centers,
                             np.zeros((k_prime,xdim))))
        start_idx = orig_k

    X_ids = np.arange(numx)

    idx = []
    idx.append(int(np.random.rand() * numx))
    
    if start_idx == 0:
        # use this if there weren't centers to begin with
        centers[0] = X[idx[-1]]
        # remove the datum
        X[idx[-1]:-1] = X[idx[-1]+1:]
        X=X[:-1]
        numx -= 1
        X_ids = X_ids[:-1]

        # min dists to the centers picked so far
        dists = np.sum((X-centers[0])**2,-1)
        probs_part = (dists/dists.sum()).cumsum()
        start_idx+= 1
    else:
        dists = np.inf * np.ones(numx,dtype=float)
        for i in xrange(start_idx):
            dists = np.minimum(dists,np.sum((X-centers[i])**2,-1))
        probs_part = (dists/dists.sum()).cumsum()

    for i in xrange(start_idx,k):
        #  get all the  centers taken care of
        idx.append(X_ids[np.random.rand() <probs_part][0])
        centers[i] = X[idx[-1]]
        # remove the datum
        if idx[-1]+1 < len(X):
            X[idx[-1]:-1] = X[idx[-1]+1:]
            dists[idx[-1]:-1] = dists[idx[-1]+1:]

        X = X[:-1]
        dists = dists[:-1]
        numx -= 1
        X_ids = X_ids[:-1]
        
        
        dists = np.minimum(dists,np.sum((X-centers[i])**2,-1))
        probs_part = (dists/dists.sum()).cumsum()

    # unwhiten X and centers
    X *= s
    X += m
    centers *= s
    centers += m
    X = X.reshape((numx,) + xshape)
    return centers.reshape((k,) + xshape)

def GMM_EM(X,centers,cov_type='diag',tol=.00001,init_sigmas=None):
    """
    Gaussian mixture modeling.  Begin by estimating a covariance matrix
    over all data to give a covariance to each center point

    Parameters:
    ===========
    X: numpy.ndarray[ndim=2]
        dimension 0 varies over data points, dimension 1 varies over
        data coordinates
    centers: numpy.ndarray[ndim=2]
        dimension 0 varies over centers points, dimension 1 varies over
        centers coordinates. These are the initial starting centers
        for the data
    cov_type: str
        'diag' - Covariances are diagonal, only option implemented presently
    
    """
    numx = X.shape[0]
    xshape = X.shape[1:]
    xdim = np.prod(xshape)
    X = X.reshape(numx,xdim)
    
    num_mix = centers.shape[0]
    centers = centers.reshape(num_mix,xdim)
    # initialize the covariances to a common covariance
    
    sigmas = np.tile(np.var((X - np.mean(centers,0)),axis=0),(num_mix,1))
    #if init_sigmas is not None:
    #    for i,sigma in enumerate(init_sigmas):
    #        sigmas[i][:] =sigma.reshape(xdim)

    # initialize uniform weights
    weights = 1./num_mix * np.ones(num_mix)

    membership_probs = np.zeros((num_mix,numx),dtype=np.float)
    max_cluster_log_likes = np.zeros(numx,dtype=np.float)
    membership_norm_constants = np.zeros(numx,dtype=np.float)




    # get the initial sufficient statistics
    log_like = e_step(X,centers,sigmas,
                                        weights,membership_probs,
                                        max_cluster_log_likes,
                                        membership_norm_constants)
    log_like_increase_ratio = np.inf
    iteration_num=0
    while log_like_increase_ratio > tol:
        m_step(X,membership_probs,centers,
                                          sigmas,weights)
        new_log_like = e_step(X,centers,sigmas,weights,
                                            membership_probs,max_cluster_log_likes,
                                            membership_norm_constants)
        log_like_increase_ratio = (log_like - new_log_like )/log_like
        log_like=new_log_like
        print  "log_like_increase_ratio: %g" % log_like_increase_ratio
        print "log-likelihood: %g" % log_like
        print "iteration: %d" % iteration_num
        iteration_num+=1
    
    
    X = X.reshape((numx,)+xshape)
    centers = centers.reshape((num_mix,) + xshape)

    sigmas = sigmas.reshape((num_mix,) + xshape)
    return centers,sigmas,weights,membership_probs.T
    

def m_step(X,membership_probs,centers,sigmas,weights):
    """
    Using the membership probabilities compute the conditionally
    maximum likelihood mean and diagonal covariance
    """
    weights[:] = membership_probs.sum(-1)

    # normalize the membership probabilities for maximization purposes
    for mix_component, weight in enumerate(weights):
        membership_probs[mix_component] /= weight


    weights /= weights.sum()

    centers[:] = np.dot(membership_probs,X)
    
    sigmas = np.dot(membership_probs,(X-centers[mix_component])**2)
        

    
def e_step(X,centers,sigmas,weights,membership_probs,
           max_cluster_log_likes,membership_norm_constants):
    """
    Computes the log-likelihood and the conditional likelihood that data point
    i was generated by class k given the value of data point i.

    Parameters:
    ==========
    X: numpy.ndarray[ndim=2]
        dimension 0 varies over data points, dimension 1 varies over
        data coordinates
    centers: numpy.ndarray[ndim=2]
        dimension 0 varies over centers points, dimension 1 varies over
        centers coordinates. These are the initial starting centers
        for the data
    sigmas: numpy.ndarray[ndim=2]
    membership_probs: numpy.ndarray[ndim=2]
    max_cluster_log_likes: numpy.ndarray[ndim=1]
        this should be a vector of length X.shape[0] which
        will be a holder to find the maximum value for the cluster
        log-likelihood (or minimum negative log-likelihood)

    Returns:
    ========
    log_like: float
        Log likelihood value
    membership_probs: numpy.ndarray[ndim=2]
        dimension 0 is the number of data points and it indexes which data
        point the membership probs are for. Dimension 1 has length num_mix
        and for a given data point it gives the conditional likelihood for
        a mixture component
        
    """
    num_mix = centers.shape[0]
    NORMAL_CONSTANT = (- centers.shape[1]/2.) * np.log(2*np.pi)

    max_cluster_log_likes[:] = -np.inf
    for mix_component, center_sigma in enumerate(itertools.izip(centers,sigmas)):
        center,sigma = center_sigma
        sigma_term = -.5 * np.log(sigma).sum()
        membership_probs[mix_component][:] = - .5 *((X-center)**2 / sigma).sum(1) + NORMAL_CONSTANT + sigma_term
        max_cluster_log_likes[:] = np.maximum(membership_probs[mix_component],
                                              max_cluster_log_likes)
        
    
    membership_probs[:] = np.exp(membership_probs - max_cluster_log_likes)
    
    membership_probs = np.dot(np.diag(weights),membership_probs)

    membership_norm_constants = membership_probs.sum(0)
    log_likelihood = (np.log(membership_norm_constants.sum(0)) + max_cluster_log_likes).sum()
    
    membership_probs /= membership_norm_constants
    return log_likelihood
                                          
        
        
        
def simple_test():
    """
    Run a simple test of the GMM
    """
    centers = np.array([[1.,1.,0,8],[-1.,10,-20,2]])
    weights = np.array([.5,.5])
    X = np.random.randn(100,4)
    for i in xrange(len(X)):
        X[i] += centers[i % 2]
    centers,sigmas,weights = GMM_EM(X,centers,cov_type='diag',tol=.00001)

    

def main(args):
    print args

    X, X_metadata = get_data_vec(args.data_dir,args.save_tag)
    np.savez('%saa_r_examples_%s.npz' % (args.data_dir,args.save_tag),X,X_metadata)
    
    if args.get_centers:
        for p in xrange(7):
            num_mix = 2**p
            print "num_mix=%d" % num_mix
            centers = kmeanspp_gmm_init(num_mix,X)
            np.save('%saa_r_centers_%d_%s.npy' % (args.data_dir,num_mix,args.save_tag),
                    centers)
    
    if args.run_simple_test:
        # generate the data
        
        simple_test()
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser("Program to do the initial gmm-clustering for training the HMM-based classifier")
    parser.add_argument("--data_dir",default="data/",type=str,
                        help="data directory where the saved data vectors are going to be drawn from and where the outputs are going to be saved to")
    parser.add_argument("--save_tag",default="train_mfcc",type=str,
                        help="an identifying suffix appended to the end of file names in order to know where they come from")
    parser.add_argument("--get_centers",action="store_true",
                        help="Whether to run the kmeans++ initialization on the vectors")
    parser.add_argument("--num_centers",default=[1],nargs='*',type=int,
                        help="number of mixture components to test")
    parser.add_argument("--run_simple_test",action="store_true",
                        help="simple test of whether the GMM algorithm is working")
    main(parser.parse_args())

    
