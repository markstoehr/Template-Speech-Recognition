import numpy as np

def simple_estimate_template(pattern_examples,template_length=None):
    if not template_length:
        template_length = int(_get_template_length(pattern_examples))
    num_examples = len(pattern_examples)
    template_height = pattern_examples[0].shape[0]
    registered_templates = _register_all_templates(num_examples,
                                                   template_height,
                                                   template_length,
                                                   pattern_examples)
    return template_height,template_length,registered_templates, np.minimum(np.maximum(np.mean(registered_templates,axis=0),.05),.95)

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


