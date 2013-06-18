import EM_loc_length
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def createRandomModels(min_lengths,

                             num_features=100,
                             num_lengths=10,
                             num_ridges=4,
                             blurring_bandwidth=2,
                             min_probability=0.01):
    """
    Creates a number of random models
    """
    num_models = len(min_lengths)
    max_example_length = num_lengths + min_lengths.max() -1
    models = np.zeros((num_models,
                       max_example_length,
                       num_features))
    for model_id, min_length in enumerate(min_lengths):
        models[model_id,:min_length+num_lengths-1] = _createRandomModel( min_length,
                                                                       num_features,
                                                                       num_lengths,
                                                                       num_ridges,
                                                                       blurring_bandwidth,
                                                                       min_probability=min_probability)
    
    
    return np.clip(models,min_probability,1-min_probability)


def _createRandomModel( min_length,
                      num_features,
                      num_lengths,
                      num_ridges,
                      
                      blurring_bandwidth,
                      min_probability=0.01):
    """

    Creates a random template using a ridge-process
    where ridges are placed at a random orientation
    in the model

    Parameters:
    ============
    min_length: int
        Minimum length for the model
    num_features: int
        Number of features for each time point
    num_lengths: int
        Indicates how long the model can be at maximum
        min_length + num_lengths
    ridge_length_parameter: float
        should be between 0 and 1
        assume that ridges have a poisson distribution over their
        lengths (add one to it, though)
        these are percentages of the length inferred by min_length+ num_lengths,
        and num_features
    blurring_bandwidth: float
        how much to blur the template
    min_probability: float
        constrain the edges of the template
    """

    max_model_length = min_length+num_lengths-1

    model = np.zeros((max_model_length,
                      num_features))
    model_xx,model_yy = np.meshgrid(np.arange(num_features),
                                    np.arange(
            max_model_length)
                                    )
    
    num_approx_terms = 1000

    for ridge_id in xrange(num_ridges):
        # get random start and end points
        ridge_start_y = np.random.randint(max_model_length)
        ridge_start_x = np.random.randint(num_features)
        
        ridge_end_y = np.random.randint(max_model_length)
        ridge_end_x = np.random.randint(num_features)
        
        
        mask  = np.abs( model_yy - (ridge_end_y - ridge_start_y)* (model_xx - ridge_start_x)/(ridge_end_x - ridge_start_x) - ridge_start_y) <= .01

# += ((( model_xx - ridge_start_x)/(ridge_end_x - ridge_start_x) >= i*1./num_approx_terms) 
#                     * (( model_xx - ridge_start_x)/(ridge_end_x - ridge_start_x) <= (i+1)*1./num_approx_terms) 
#                     * (( model_yy - ridge_start_y)/(ridge_end_y - ridge_start_y) <= (i+1)*1./num_approx_terms) 
#                     * (( model_yy - ridge_start_y)/(ridge_end_y - ridge_start_y) >= i*1./num_approx_terms) )
        try:
            model[mask] = 1.
        except:
            import pdb; pdb.set_trace()
        
    
    return np.clip(gaussian_filter(model,blurring_bandwidth),min_probability,1-min_probability)
    
        
def createRandomData(models,
                     num_examples,
                     min_lengths,
                     num_lengths,
                     num_start_times,
                     background_model,):
    """
    Create random data with a given model and examples
    We assume that num_start_times is odd so that there
    is a middle start time
    """
    num_features = models.shape[2]
    max_example_length = num_start_times-1+min_lengths.max() + num_lengths-1
    examples = np.zeros((num_examples,
                         max_example_length,
                         num_features),
                        dtype=np.uint8)
    example_start_times = (num_start_times-1)/2* np.ones(num_examples,dtype=int)
    for i in xrange(num_examples):
        model_id = i % len(models)
        start_time = np.random.randint(num_start_times)
        example_start_times[i] = start_time
        add_length = np.random.randint(num_lengths)
        if start_time > 0:

            examples[i,:start_time] = (np.random.rand(start_time,
                                                      num_features) <= background_model).astype(np.uint8)
        
        examples[i,start_time:start_time+min_lengths[model_id]+add_length] = (
            np.random.rand(min_lengths[model_id]+add_length,
                            num_features) <= models[model_id,:min_lengths[model_id]+add_length]).astype(np.uint8)
        
        model_end_time = start_time+min_lengths[model_id] + add_length
        if model_end_time < max_example_length:
            examples[i,model_end_time:] = (
                np.random.rand(max_example_length-model_end_time,
                                num_features) <= background_model).astype(np.uint8)
    
    return examples,example_start_times
                         
    
def testSimpleComputeLogLikelihood():
    np.random.seed(0)
    min_lengths = np.array((20,))
    num_features=100
    num_lengths=1
    num_ridges=4
    min_probability=0.01
    M = createRandomModels(min_lengths=min_lengths,
                           num_features=num_features,
                           num_lengths=num_lengths,
                           num_ridges=num_ridges,
                           blurring_bandwidth=2,
                           min_probability=min_probability
                           )
    log_M = np.log(M)
    log_inv_M = np.log(1-M)

    num_models = len(M)

    background_model = M.min() * np.ones(num_features)
    # generate the data set
    # num_start_times specifies how many start times are allowed
    # for the models
    num_start_times = 1
    num_examples=10
    E,example_labeled_start_times = createRandomData(models=M,
                         num_examples=num_examples,
                         min_lengths=min_lengths,
                         num_lengths=num_lengths,
                         num_start_times=num_start_times,
                         background_model=background_model,
                         )
    
    L = np.zeros((num_examples,num_models,num_start_times,
                  num_lengths))

    max_loglikes_by_example = np.zeros(num_examples)

    

    (E_bgd_frame_scores,
     E_bgd_scores_prefix,
     E_all_bgd_scores) = computeBackgroundLogLikelihoods(E,np.log(background_model),np.log(1-background_model))

    log_weights = np.log(np.ones((num_models,num_lengths)) * 1./(num_models * num_lengths))

    computeLogLikelihood(E,L,
                         min_lengths,
                         example_labeled_start_times,
                         log_weights,
                         log_M,
                         log_inv_M,
                         max_loglikes_by_example, 
                         E_bgd_frame_scores,
                       E_bgd_scores_prefix,E_all_bgd_scores)

    example_lls = np.zeros((num_examples,num_lengths))
    L[:] = 0.
    out_ll = EStep(E,L,min_lengths,
                   example_labeled_start_times,log_weights,log_M,
                   log_inv_M,max_loglikes_by_example, E_bgd_frame_scores,
                   E_bgd_scores_prefix,E_all_bgd_scores,example_lls)

    

def testComputeLogLikelihood():
    # create a test

    np.random.seed(0)
    min_lengths = np.array((20,))
    num_features=100
    num_lengths=10
    num_ridges=4
    M = createRandomModels(min_lengths=min_lengths,
                           num_features=num_features,
                           num_lengths=num_lengths,
                           num_ridges=num_ridges,
                           blurring_bandwidth=2,
                           min_probability=0.01
                           )
    log_M = np.log(M)
    log_inv_M = np.log(1-M)

    num_models = len(M)

    background_model = M.min() * np.ones(num_features)
    # generate the data set
    # num_start_times specifies how many start times are allowed
    # for the models
    num_start_times = 5
    num_examples=10
    E,example_labeled_start_times = createRandomData(models=M,
                         num_examples=num_examples,
                         min_lengths=min_lengths,
                         num_lengths=num_lengths,
                         num_start_times=num_start_times,
                         background_model=background_model,
                         )
    
    L = np.zeros((num_examples,num_models,num_start_times,
                  num_lengths))

    max_loglikes_by_example = np.zeros(num_examples)

    

    (E_bgd_frame_scores,
     E_bgd_scores_prefix,
     E_all_bgd_scores) = computeBackgroundLogLikelihoods(E,np.log(background_model),np.log(1-background_model))

    log_weights = np.log(np.ones((num_models,num_lengths)) * 1./(num_models * num_lengths))

    computeLogLikelihood(E,L,
                         min_lengths,
                         example_labeled_start_times,
                         log_weights,
                         log_M,
                         log_inv_M,
                         max_loglikes_by_example, 
                         E_bgd_frame_scores,
                       E_bgd_scores_prefix,E_all_bgd_scores)
    
    EStep(E,L,min_lengths,
           example_labeled_start_times,log_M,
           log_inv_M,max_loglikes_by_example, mean_log_like_bgd_model,
           E_bgd_scores_prefix,E_all_bgd_scores)
    
    
