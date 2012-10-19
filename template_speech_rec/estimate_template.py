import numpy as np
import itertools

def pad_examples_bgd_samples(examples,lengths,bgd_probs):
    max_length = examples.shape[1]
    out = []
    for example, length in itertools.izip(examples,lengths):
        diff = max_length - length
        if diff >0 :
            out.append(
                np.vstack((example[:length],
                       (np.random.rand(diff,
                                      examples.shape[2],
                                       examples.shape[3]) > np.tile(bgd_probs,(diff,1,1))).astype(np.uint8))))
        else:
            out.append(example)
    return np.array(out).astype(np.uint8)

def recover_different_length_templates(affinities,examples,lengths):
    affinities_trans = affinities.T
    affinities_trans /= affinities_trans.sum(1)[:,np.newaxis]
    avg_lengths = (np.dot(affinities_trans,lengths)  + .5).astype(int)
    example_shapes = examples.shape[1:]
    return tuple(
        template[:length] 
        for template,length in itertools.izip(np.dot(affinities_trans,examples.reshape(examples.shape[0],
                                                                                       np.prod(example_shapes))).reshape((avg_lengths.shape[0],)+example_shapes),avg_lengths))

        
    
            

def _register_template_time_zero(T,template_time_length):
    """
    T is the current example, we work with the time axis
    being possibly anywhere, we assume that the time
    axis is along the zeroth dimension
    """
    example_time_length = float(T.shape[0])
    template_time_length = float(template_time_length)
    return T[np.clip(((np.arange(template_time_length)/template_time_length) * example_time_length + .5).astype(int),0,int(T.shape[0]-1))]

def register_templates_time_zero(examples,lengths,min_prob=.01):
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


