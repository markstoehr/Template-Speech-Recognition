import numpy as np
import itertools

def extend_example_to_max(syllable_example,clipped_bgd,max_length):
    if syllable_example.shape[0] >= max_length:
        return syllable_example.astype(np.uint8)
    else:
        return np.vstack((syllable_example,
                          (np.random.rand(max_length-syllable_example.shape[0],
                                          1,1) <= np.tile(clipped_bgd,
                                  (max_length-syllable_example.shape[0],1,1))).astype(np.uint8))).astype(np.uint8)



def extend_examples_to_max(clipped_bgd,syllable_examples,lengths=None,
                           return_lengths=False):
    if lengths is None:
        lengths = np.array(tuple(syllable_example.shape[0]
                                 for syllable_example in syllable_examples))
    max_length = lengths.max()
    if return_lengths:
        return (np.array(tuple(
            extend_example_to_max(syllable_example,clipped_bgd,max_length)
            for syllable_example in syllable_examples)),
            lengths)
    else:
        return np.array(tuple(
            extend_example_to_max(syllable_example,clipped_bgd,max_length)
            for syllable_example in syllable_examples))


def pad_examples_bgd_samples(examples,lengths,bgd_probs):
    max_length = examples.shape[1]
    out = np.zeros(examples.shape,dtype=np.uint8)
    for idx,v in enumerate(itertools.izip(examples,lengths)):
        example, length  = v
        diff = max_length - length
        if diff >0 :
            out[idx] = np.vstack((example[:length],
                       (np.random.rand(diff,
                                      examples.shape[2],
                                       examples.shape[3]) <= np.tile(bgd_probs,(diff,1,1))).astype(np.uint8)))
        else:
            out[idx][:] = example
    return out

def recover_different_length_templates(affinities,examples,lengths,
                                       block_size=5000,
                                       do_truncation=True,sigmas=None):
    """
    Now uses a memory efficient algorithm where block_size
    is the most of the matrix that is used at any given time

    The algorithm works as follows: we keep running estimates of the
    affinity sums for each template affinity in affinity sums we also
    keep running template averages, these are the basic estimates that
    then get updated iteratively as we go through chunks of the data
    at a time.


    This algorithm has been verified to work

    """
    affinities_trans = affinities.T
    affinities_trans /= affinities_trans.sum(1)[:,np.newaxis]
    if do_truncation:
        avg_lengths = (np.dot(affinities_trans,lengths)  + .5).astype(int)
    else:
        avg_lengths = np.ceil(np.dot(affinities_trans,lengths)).astype(int)
        max_length = lengths.max()
        for i in xrange(avg_lengths.shape[0]):
            avg_lengths[i]=max_length

    if sigmas is not None:
        out_sigmas = tuple(
            sigma[:length]
            for sigma,length in itertools.izip(sigmas,avg_lengths))

    example_shapes = examples.shape[1:]
    num_data = examples.shape[0]
    if num_data < block_size:

        out_templates =  tuple(
            template[:length]
            for template,length in itertools.izip(np.dot(affinities_trans,examples.reshape(examples.shape[0],
                                                                                           np.prod(example_shapes))).reshape((avg_lengths.shape[0],)+example_shapes),avg_lengths))

        if sigmas is not None:
            return out_templates, out_sigmas
        else:
            return out_templates
    else:
        for cur_chunk in xrange(num_data/block_size):
            # print "Working on chunk %d" % cur_chunk
            if cur_chunk == 0:
                template_estimates = [
                    template[:length]
                    for template,length in itertools.izip(np.dot(affinities_trans[:,:block_size],examples[:block_size].reshape(block_size,
                                                                                                                               np.prod(example_shapes))).reshape((avg_lengths.shape[0],)+example_shapes),avg_lengths)]
            else:
                start_idx = cur_chunk*block_size
                cur_block_size = min(block_size,
                                     num_data - start_idx)
                end_idx = start_idx + cur_block_size

                new_template = np.dot(
                    affinities_trans[:,start_idx:end_idx],
                    examples[start_idx:end_idx].reshape(
                        block_size,
                        np.prod(example_shapes))
                    ).reshape((avg_lengths.shape[0],)+example_shapes)

                for i, length in enumerate(avg_lengths):
                    template_estimates[i] += new_template[i][:length]
        if sigmas is not None:
            return template_estimates,out_sigmas
        else:
            return template_estimates




def recover_clustered_data(affinities,padded_examples,templates,assignment_threshold = .95):
    row_sums = affinities.sum(1)
    template_assignments = (affinities/row_sums[:,np.newaxis] > assignment_threshold).astype(int).T
    cluster_sizes = template_assignments.sum(1)
    return tuple(
        padded_examples[template_assignments[i]==1][:,:templates[i].shape[0]]
        for i in xrange(len(templates)))




def _register_template_time_zero(T,template_time_length):
    """
    T is the current example, we work with the time axis
    being possibly anywhere, we assume that the time
    axis is along the zeroth dimension
    """
    example_time_length = float(T.shape[0])
    template_time_length = float(template_time_length)
    return T[np.clip(((np.arange(template_time_length)/template_time_length) * example_time_length + .5).astype(int),0,int(T.shape[0]-1))]

def register_templates_time_zero(examples,lengths=None,min_prob=.01):
    if lengths is None:
        lengths = np.array([len(e) for e in examples])
    mean_length = int(lengths.mean()+.5)
    registered_examples = np.array(
        tuple(
            _register_template_time_zero(example[:length],mean_length)
            for example,length in itertools.izip(examples,lengths)))
    return np.clip(registered_examples.mean(0),
                   min_prob,
                   1-min_prob), registered_examples

def construct_linear_filters(Ts,
                            bgd,all_cs=False,use_spectral=False,T_sigmas=None,bgd_sigma=None,min_prob=.01):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    all_cs option returns many different cs corresponding to different lengths of the template
    """
    if use_spectral:
        return tuple(
            construct_spectral_linear_filter(T,bgd,T_sigma,bgd_sigma,all_cs=all_cs)
            for T,T_sigma in itertools.izip(Ts,T_sigmas))
    else:
        return tuple(
            construct_linear_filter(np.clip(T,min_prob,
                                            1-min_prob),
                                    np.clip(bgd,min_prob,
                                            1-min_prob),all_cs=all_cs,use_spectral=use_spectral)
            for T in Ts)


def construct_linear_filters_contiguous(models,
                            bgd,use_spectral=False,T_sigmas=None,bgd_sigma=None,min_prob=.01):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    all_cs option returns many different cs corresponding to different lengths of the template
    """
    if use_spectral:
        return tuple(
            construct_spectral_linear_filter(T,bgd,T_sigma,bgd_sigma,all_cs=all_cs)
            for T,T_sigma in itertools.izip(Ts,T_sigmas))
    else:
        filters = np.zeros(models.shape)
        biases = np.zeros((models.shape[0],models.shape[1]))
        for T_idx, T in enumerate(models):
            filters[T_idx], biases[T_idx] = construct_linear_filter(np.clip(T,min_prob,
                                            1-min_prob),
                                    np.clip(bgd,min_prob,
                                            1-min_prob),all_cs=True,use_spectral=use_spectral)
        return filters, biases



def construct_spectral_linear_filter(T,bgd,T_sigma,bgd_sigma,all_cs=False,save_as_type=np.float32):
    """
    Spectral linear filter assumed to be using diagonal covariance matrices
    also the background sigma only models a single frame where as the T_sigmas
    model different numbers of frames, this has to be taken in to account
    for constructing the filter

    Parameters:
    ===========
    T: numpy.ndarray[ndim=2]
        Same dimension as X, mean of the template
    T_sigma: numpy.ndarray[ndim=2]
        Same dimension as X, var of each coordinate
        for the template, essentially a diagonal
        covariance matrix
    bgd: numpy.ndarray[ndim=1]
        Single frame of background means
    bgd_sigma: numpy.ndarray[ndim=1]
        single frame of backgroundvariances

    """
    num_frames = T.shape[0]
    constant = - .5 * np.log(T_sigma).sum() + num_frames * np.log(bgd_sigma).sum()/2.
    inv_bgd_sigma = bgd_sigma**-1
    inv_T_sigma = T_sigma**-1
    second_moment_filter = -.5 * (inv_T_sigma - inv_bgd_sigma)
    first_moment_filter = T * inv_T_sigma - (bgd * inv_bgd_sigma)
    constant += (- .5 * (T**2 * inv_T_sigma - (bgd**2 * inv_bgd_sigma))).sum()

    return (second_moment_filter.astype(save_as_type),first_moment_filter.astype(save_as_type),save_as_type(constant))



def construct_linear_filter(T,
                            bgd,min_prob=.01,all_cs=False,
                            use_spectral=False,sigma=None):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    """

    T = np.clip(T,min_prob,1-min_prob)
    Bgd = np.tile(bgd,
                  (T.shape[0],) +tuple( np.ones(len(T.shape)-1)))
    T_inv = 1. - T
    Bgd_inv = 1. - Bgd
    C_exp_inv = T_inv/Bgd_inv
    if all_cs:
        c = np.cumsum(np.log(C_exp_inv.reshape(*((len(C_exp_inv),) +
                                           (np.prod(C_exp_inv.shape[1:]),)))).sum(1))
        
    else:
        c = np.log(C_exp_inv).sum()
    expW = (T/Bgd) / C_exp_inv

    return np.log(expW).astype(np.float32), c

def construct_likelihood_linear_filters(Ts,min_prob=.01,all_cs=False,
                            use_spectral=False,sigma=None):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    """

    return tuple(
        construct_likelihood_linear_filter(T,min_prob=min_prob,all_cs=all_cs)
        for T in Ts)
                                           

def construct_likelihood_linear_filter(T,min_prob=.01,all_cs=False,
                            use_spectral=False,sigma=None):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    """

    T = np.clip(T,min_prob,1-min_prob)
    T_inv = 1. - T
    C_exp_inv = T_inv
    if all_cs:
        c = np.cumsum(np.log(C_exp_inv.reshape(len(C_exp_inv),
                                               np.prod(C_exp_inv.shape[1:]))).sum(1)[::-1])[::-1]
    else:
        c = np.log(C_exp_inv).sum()
    expW = T / C_exp_inv

    return np.log(expW).astype(np.float32), c


def construct_linear_filter_structured_alternative(T1,T2,
                            bgd=None,min_prob=.01):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    """
    if bgd is None:
        bgd = .5 * np.ones(T1.shape[1:])
    T1 = np.clip(T1,min_prob,1-min_prob)
    T2 = np.clip(T2,min_prob,1-min_prob)
    if T1.shape[0] < T2.shape[0]:
        T1p = np.vstack((T1,
                         np.tile(bgd,
                  (T2.shape[0] - T1.shape[0],) + tuple(1 for i in xrange(len(T1.shape)-1)))))
        T2p = T2
    elif T2.shape[0] < T1.shape[0]:
        T2p = np.vstack((T2,
                         np.tile(bgd,
                  (T1.shape[0] - T2.shape[0],) + tuple(1 for i in xrange(len(T1.shape)-1)))))
        T1p = T1
    else:
        T1p = T1
        T2p = T2
    T1_inv = 1. - T1p
    T2_inv = 1. - T2p
    C_exp_inv = T1_inv/T2_inv
    c = np.log(C_exp_inv).sum()
    expW = (T1p/T2p) / C_exp_inv
    return np.log(expW).astype(np.float32), c


def simple_estimate_template(pattern_examples,template_length=None):
    if not template_length:
        template_length = int(_get_template_length(pattern_examples))
    num_examples = len(pattern_examples)
    template_height = pattern_examples[0].shape[0]
    registered_templates = _register_all_templates(num_examples,
                                                   template_height,
                                                   template_length,
                                                   pattern_examples)
    return template_height,template_length,registered_templates, np.clip(np.mean(registered_templates,axis=0),.05,.95)

def get_template_subsample_mask(T,threshold):
    """
    Make a coarsened version of the template
    no need to do any spreading since that
    has already been done, hence we just need to subsample
    """
    return T > threshold


def _register_all_templates(num_examples,
                            template_height,
                            template_length,
                            pattern_examples):
    registered_templates = np.zeros((num_examples,
                                     template_height,
                                     template_length))
    for pe_idx in xrange(len(pattern_examples)):
        _register_template(pattern_examples[pe_idx],
                           registered_templates[pe_idx,:,:],
                           template_height,
                           template_length)
    return registered_templates


def _get_template_length(pattern_examples):
    """ Computes the median length of the pattern examples
    """
    return np.median(np.array(\
            map(lambda E:E.shape[1],
                pattern_examples)))

def _register_template(T,new_template,template_height,template_length):
    example_length = T.shape[1]
    if example_length < template_length:
        for frame_idx in xrange(template_length):
            mapped_idx = int(frame_idx/float(template_length) *example_length)
            new_template[:,frame_idx] = T[:,mapped_idx]
    else:
        for frame_idx in xrange(example_length):
            mapped_idx = int(frame_idx/float(example_length)*template_length)
            new_template[:,mapped_idx] = np.maximum(new_template[:,mapped_idx],
                                                    T[:,frame_idx])


def convert_to_neg_template(P,W):
    return 1./(1. + (1./P -1)*np.exp(W))

def convert_to_pos_template(Q,W):
    return 1./(1. + (1./Q -1)/np.exp(W))

def convert_to_pos_template_uniform_mixture(Ps,Qs,Ws,clip_factor=.0001):
    """
    Neg template has the same probabilities across time
    so it is a uniform template, and also across different mixture
    components

    Uses the linear filters and the negative templates to infer the
    positive templates

    Parameters:
    ===========
    Ps:
        Tuple of arrays where each array has entries in (0,1)
        Time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        frequency location
    Qs:
        Tuple of arrays where each array has entries in (0,1)
        should be uniform across the time axis
        Time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        frequency location
    Ws:
        Tuple of arrays where each array has entries in (0,1)
        SVM filter, time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        time-frequency location

    clip_factor:
        In order for the templates to work, the probabilities
        need to greater than zero and less than 1 (for logarithms)
        so we give an alpha such that all probabilities are
        in the interval [clip_factor, 1-clip_factor]

    """
    for i,QW in enumerate(itertools.izip(Qs,Ws)):
        Q,W = QW
        Ps[i][:] = np.clip(1./(1. + (1./Q -1)/np.exp(W)),clip_factor,
                        1-clip_factor)

    return Ps


def convert_to_neg_uniform_template(P,W):
    """
    Neg template has the same probabilities across time
    so it is a uniform template

    Parameters:
    ===========
    P:
        Time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        frequency location
    W:
        SVM filter, time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        time-frequency location


    """
    num_time_frames = P.shape[0]
    # sum(0) sums over the time axis
    return np.tile((1./(1. + (1./P -1)*np.exp(W))).sum(0)/num_time_frames,
                   (num_time_frames,1,1))

def convert_to_neg_uniform_template_mixture(Ps,Qs,Ws,clip_factor=.0001):
    """
    Neg template has the same probabilities across time
    so it is a uniform template, and also across different mixture
    components

    Finds a single vector that minimizes the difference in terms
    of squared error from Ws and Ps

    Parameters:
    ===========
    Ps:
        Tuple of arrays where each array has entries in (0,1)
        Time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        frequency location
    Qs:
        Tuple of arrays where each array has entries in (0,1)
        should be uniform across the time axis
        Time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        frequency location
    Ws:
        Tuple of arrays where each array has entries in (0,1)
        SVM filter, time axis corresponds to dimension 0
        dimension 1 corresponds to frequency channels
        dimension 2 corresponds to features at that
        time-frequency location

    clip_factor:
        In order for the templates to work, the probabilities
        need to greater than zero and less than 1 (for logarithms)
        so we give an alpha such that all probabilities are
        in the interval [clip_factor, 1-clip_factor]
    """
    # implement the online averaging tool to get
    # the average positive template
    q = np.zeros(Ps[0].shape[1:])
    total_time_frames = 0
    for P,W in itertools.izip(Ps,Ws):
        num_time_frames = P.shape[0]
        total_time_frames += num_time_frames
        q+= ((1./(1. + (1./P -1)*np.exp(W))).sum(0) - num_time_frames*q)/total_time_frames


    q = np.clip(q,clip_factor,1-clip_factor)
    for i in xrange(len(Qs)):
        Qs[i][:] = q

    return Qs



def iterative_neg_template_estimate(svmW,P,svmC,
                                    max_iter = 1000,
                                    tol=.00001,
                                    verbose=False):
    scaling = 1.
    W = scaling * svmW
    C = scaling * svmC
    lfW = np.zeros(W.shape)
    cur_error = abs(np.sum(scaling*W - lfW))/abs(scaling*W.sum())
    num_iter = 0
    while cur_error > tol and num_iter < max_iter:
        Q = np.clip(convert_to_neg_template(P,scaling*svmW),.0001,1-.0001)
        lfC = np.sum(np.log((1-P)/(1-Q)))
        scaling = lfC/C
        lfW = np.log(P/(1.-P) * (1.-Q)/Q)
        cur_error = abs(np.sum(scaling*W - lfW))/abs(scaling*W.sum())
        if verbose:
            print "lfC=%g\tnorm(lfW)=%g\tsvmC=%g\tcur_error=%g\tscaling=%g" % (lfC,np.linalg.norm(lfW),svmC,cur_error,scaling)
        num_iter += 1
    print "lfC=%g\tnorm(lfW)=%g\tsvmC=%g\tcur_error=%g\tscaling=%g" % (lfC,np.linalg.norm(lfW),svmC,cur_error,scaling)
    return lfW,Q,lfC

def make_Qs_from_Ps_bgd(Ps,bgd):
    """
    Parameters:
    ===========
    Ps: tuple of arrays
        For each array in the tuple the
        zeroeth dimension is the time dimension
        the other dimensions are features and the shapes
        are the same between each P array except
        for the time dimension. The Q arrays will
        have the same number of time points as their
        corresponding P array
    bgd: array
        the shape of this array should be equal to P.shape[1:]
        for each P array in Ps, the Q array in Qs will simply
        be stacked copies of these arrays

    Returns:
    ========
    Qs: tuple of Arrays
        each Q array is simply a sequence of the bgd vectors
        corresponding to the length of the P array
    """
    return tuple(
        np.tile(bgd.ravel(),P.shape[0]).reshape((P.shape[0],) + bgd.shape)
        for P in Ps)

def make_lfWs(Ps,Qs):
    """
    Parameters:
    ===========
    Ps: tuple of arrays
        Positive templates
    Qs: tuple of arrays
        negative templates

    Returns:
    ========
    lfWs: tuple of arrays
    """
    return tuple(
        np.log(P/(1.-P)*(1.-Q)/Q)
        for P,Q in itertools.izip(Ps,Qs))

def get_lfCs(Ps,Qs):
    """
    Parameters:
    ===========
    Ps: tuple of template arrays
    Qs: tuple of template arrays

    """
    return np.array(tuple(np.sum(np.log((1-P)/(1-Q)))
                 for P,Q in itertools.izip(Ps,Qs)))

def get_lfWs_svmWs_error(lfWs,Ws):
    """
    Compute the squared error deviation

    Parameters:
    ===========
    lfWs: tuple of arrays
        Each array in this tuple is formed from the
        linear filter implied by a positive template P
        and a negative template Q
    Ws: tuple of arrays
        These are the scaled SVM templates implied by the array
    """
    total_error_vec = np.array(tuple(
            (np.linalg.norm(lfW - W)**2,np.linalg.norm(W)**2)
            for lfW,W in itertools.izip(lfWs,Ws)))
    a,b=np.sqrt(total_error_vec.sum(0))
    individual_errors = np.sqrt(total_error_vec[:,0]/total_error_vec[:,1])
    return a/b, individual_errors

def get_scaled_Ws(scalings,svmWs):
    """
    Parameters:
    ===========
    scalings: numpy.ndarray[ndim=1]
        Scaling for each of the linear filters and the svm components
    svmWs: tuple of numpy.ndarray[ndim=3]
        Tuple of the linear filters learned using
        the SVM

    Returns:
    ========
    Ws: tuple of numpy.ndarray[ndim=3]
    """
    return tuple(scaling*W for scaling, W in itertools.izip(scalings,svmWs))




def iterative_neg_pos_templates_estimate_mixture(svmWs,svmCs,Ps,bgd,
                                    max_iter = 1000,
                                    tol=.00001,
                                    verbose=False):
    num_mix = len(svmWs)
    scalings = np.ones(num_mix,dtype=float)
    Ws = get_scaled_Ws(scalings,svmWs)
    Cs = scalings * svmCs
    Qs = make_Qs_from_Ps_bgd(Ps,bgd)
    lfWs = make_lfWs(Ps,Qs)
    lfCs = get_lfCs(Ps,Qs)

    cur_error,individual_errors = get_lfWs_svmWs_error(lfWs,Ws)



    num_iter = 0
    while cur_error > tol and num_iter < max_iter:
        scalings = np.clip(lfCs/svmCs,.01,np.inf)
        Ws = get_scaled_Ws(scalings,svmWs)
        Qs = convert_to_neg_uniform_template_mixture(Ps,Qs,Ws)

        q = Qs[0][0]
        Ps = convert_to_pos_template_uniform_mixture(Ps,Qs,Ws,clip_factor=.0001)

        lfCs = get_lfCs(Ps,Qs)
        lfWs = make_lfWs(Ps,Qs)
        cur_error,individual_errors = get_lfWs_svmWs_error(lfWs,Ws)

        if verbose:
            print "lfCs=%s\nsvmC=%g\tcur_error=%g\nscalings=%s" % (str(lfCs),svmCs,cur_error,str(scalings))
        num_iter += 1
    print "lfCs=%s\nsvmC=%s\tcur_error=%g\nscalings=%s" % (str(lfCs),str(svmCs),cur_error,str(scalings))
    return lfWs,lfCs,Ps,q
