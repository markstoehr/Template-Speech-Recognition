import numpy as np
import itertools

def extend_example_to_max(syllable_example,clipped_bgd,max_length):
    if syllable_example.shape[0] >= max_length:
        return syllable_example.astype(np.uint8)
    else:
        return np.vstack((syllable_example,
                          (np.random.rand(max_length-syllable_example.shape[0],
                                          1,1) > np.tile(clipped_bgd,
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
                                       examples.shape[3]) > np.tile(bgd_probs,(diff,1,1))).astype(np.uint8)))
        else:
            out[idx][:] = example
    return out

def recover_different_length_templates(affinities,examples,lengths,
                                       block_size=5000,
                                       do_truncation=True):
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
        max_length = avg_lengths.max()
        for i in xrange(avg_lengths.shape[0]):
            avg_lengths[i]=max_length
    
    example_shapes = examples.shape[1:]
    num_data = examples.shape[0]
    if num_data < block_size:
        return tuple(
            template[:length]
            for template,length in itertools.izip(np.dot(affinities_trans,examples.reshape(examples.shape[0],
                                                                                           np.prod(example_shapes))).reshape((avg_lengths.shape[0],)+example_shapes),avg_lengths))
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
                            bgd):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    """
    return tuple(
        construct_linear_filter(T,bgd)
        for T in Ts)


def construct_linear_filter(T,
                            bgd,min_prob=.01):
    """
    Bgd is the tiled matrix of bgd vectors slaooed onto each other
    """
    T = np.clip(T,min_prob,1-min_prob)
    Bgd = np.tile(bgd,
                  (T.shape[0],1,1))
    T_inv = 1. - T
    Bgd_inv = 1. - Bgd
    C_exp_inv = T_inv/Bgd_inv
    c = np.log(C_exp_inv).sum()
    expW = (T/Bgd) / C_exp_inv
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
                  (T2.shape[0] - T1.shape[0],) + tuple(1 for i in xrange(len(T1.shape[0]-1))))))
        T2p = T2
    elif T2.shape[0] < T1.shape[0]:
        T2p = np.vstack((T2,
                         np.tile(bgd,
                  (T1.shape[0] - T2.shape[0],) + tuple(1 for i in xrange(len(T1.shape[0]-1))))))
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


