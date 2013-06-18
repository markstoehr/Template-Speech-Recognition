#!/usr/bin/python

import numpy as np

def computeBackgroundLogLikelihoods(E,log_background_model,log_inv_background_model):
    """
    Compute the background

    Out:
    ===================
    E_bgd_frame_scores[example_id,frame_id] gives the likelihood for E[example_id,frame_id] under the background model
    E_bgd_scores_prefix[example_id,frame_id] gives the likelihood for frame E[example_id, 0],\ldots, E[example_id,frame_id] under the background model
    """
    num_examples = E.shape[0]
    max_example_length = E.shape[1]
    num_features = E.shape[2]

    E_bgd_frame_scores = np.zeros((num_examples,
                                   max_example_length))
    E_bgd_scores_prefix = np.zeros((num_examples,
                                   max_example_length))
    E_all_bgd_scores = np.zeros(num_examples)
    background_log_odds = np.zeros(num_features)

    for feature_id in xrange(num_features):
        background_log_odds[feature_id] = log_background_model[feature_id] - log_inv_background_model[feature_id]

    for example_id in xrange(num_examples):
        for frame_id in xrange(max_example_length):
            E_bgd_frame_scores[example_id,frame_id] = 0
            for feature_id in xrange(num_features):                
                E_bgd_frame_scores[example_id,frame_id] += E[example_id,frame_id,feature_id] * background_log_odds[feature_id] + log_inv_background_model[feature_id]
                
            if frame_id == 0:
                E_bgd_scores_prefix[example_id,frame_id] = E_bgd_frame_scores[example_id,frame_id]
            else:
                E_bgd_scores_prefix[example_id,frame_id] = E_bgd_frame_scores[example_id,frame_id] + E_bgd_scores_prefix[example_id,frame_id-1]
            
        E_all_bgd_scores[example_id] = E_bgd_scores_prefix[example_id,max_example_length-1]
    
    return (E_bgd_frame_scores,
            E_bgd_scores_prefix,
            E_all_bgd_scores)



def computeLogLikelihood(E,L,min_lengths,example_labeled_start_times,log_weights,log_M,
                       log_inv_M,max_loglikes_by_example, E_bgd_frame_scores,
                       E_bgd_scores_prefix,E_all_bgd_scores):
    """
        We compute the log-likelihood for each data point
        under all the possible models
    L which has type ``
    
    length_maps[model_id] is a 1d array with the lengths for the maps
    length_maps[model_id][-1] is the maximum length for the model model_id
    
    Parameters:
    ===========
    L: numpy.array[ndim=4,dtype=np.float32]
        The array containing the log-likelihood statistics for different
        parameter settings.
        `L.shape[0] == num_examples` which is the number of examples
        `L.shape[1] == num_models` -- number of models
        `L.shape[2] == num_times`  -- number of start times (assumed to be odd) so that (num_times-1)/2 is the labeled starting time
                                      and we allow (num_times-1)/2 extra starting times on either side
        `L.shape[3] == num_lengths` -- number of variations in the length of the template considered

        
    example_labeled_start_times: numpy.ndarray[ndim=1,dtype=int]
        example_labeled_start_times[example_id] should be the start time corresponding to the likelihood
        for L[example_id,:,(num_times-1)/2,:]
    time the sufficient statistic for starting at time
    t+ example_labeled_start_time[example_id]
    is S[t].

    data E are assumed to be np.ndarray[ndim=3,dtype=np.uint8]
    dimension 0 is data identity
    dimension 1 is time
    dimension 2 are the features
    they should otherwise all be the same length
    
    M are the models  np.ndarray[ndim=3,dtype=np.float]
    dimension 0 is the model identity
    dimension 1 is time
    dimension 2 are features


    E_bgd_scores_prefix   --   np.ndarray[ndim=2,dtype=np.float]
    E_bgd_scores_prefix[example_id,time_t] is the sum for time =0,\ldots,time_t if background is assigned to there
    E_all_bgd_scores[example_id] - E_bgd_scores_prefix[example_id,time_t] is the sum over the scores time_t+1,\ldots, num_times
    
    
    E_all_bgd_scores -- np.ndarray[ndim=1,dtype=np.float]
    E_all_bgd_scores[example_id] is the score if the whole data point is background

    """
    num_examples, example_length, num_features = E.shape
    num_examples, num_models, num_times, num_lengths = L.shape
    
    # check the invariant
    assert num_times % 2 == 1



    for example_id in range(num_examples):
        # first possible start time for the example
        absolute_start_time = example_labeled_start_times[example_id] - (num_times-1)/2

        for model_id in range(num_models):
            # loop over the different possible start times
            for time_id in range(num_times):
                start_time = absolute_start_time + time_id
                for length_id in range(num_lengths):
                    use_length = min_lengths[model_id] + length_id
                    end_time = start_time + use_length
                    if start_time < 0:
                        L[example_id,model_id,time_id,length_id] = - np.inf
                    elif end_time > example_length:
                        L[example_id,model_id,time_id,length_id] = - np.inf
                    elif length_id == 0:
                        # here we do the whole computation for the template
                        L[example_id,model_id,time_id,length_id] = log_weights[model_id,length_id]
                        # handle the prefix background score
                        if start_time > 0:
                            # if start_time == 0:
                            #    There are no background frames at the beginning of the utterance
                            # else
                            #    there is a background frame whose last index is start_time-1
                            L[example_id,model_id,time_id,length_id] += E_bgd_scores_prefix[example_id,
                                                                 start_time-1]
                        # handle the suffix background score
                        # the frames being modeled are
                        # E[example_id,end_time], E[example_id,end_time+1], E[example_id,end_time+2], \ldots, E[example_id,T_max-1]
                        L[example_id,model_id,time_id,length_id] += E_all_bgd_scores[example_id] - E_bgd_scores_prefix[example_id,end_time-1]
                        
                        for t in range(use_length):
                            for f in range(num_features):
                                L[example_id,model_id,time_id,length_id] += E[example_id,start_time+t,f] *( log_M[model_id,t,f] - log_inv_M[model_id,t,f]) + log_inv_M[model_id,t,f]
                    else:
                        # just update the trailing suffix
                        L[example_id,model_id,time_id,length_id] = L[example_id,model_id,time_id,0] - E_bgd_frame_scores[example_id,use_length-1]
                        for f in range(num_features):
                            L[example_id,model_id,time_id,length_id] += E[example_id,start_time + use_length-1,f] *( log_M[model_id,t,f] - log_inv_M[model_id,use_length-1,f]) + log_inv_M[model_id,use_length-1,f]
                    
                    if model_id == 0 and time_id == 0:
                        max_loglikes_by_example[example_id] = L[example_id,model_id,time_id,length_id]
                    else:
                        max_loglikes_by_example[example_id] = max(
                            max_loglikes_by_example[example_id],
                            L[example_id,model_id,time_id,length_id])
    



def EStep(E,L,min_lengths,
           example_labeled_start_times,log_weights,log_M,
           log_inv_M,max_loglikes_by_example,E_bgd_frame_scores,
           E_bgd_scores_prefix,E_all_bgd_scores,example_lls):
    """
    We compute the expected value of the sufficient statistics
    The sufficient statistics are a four-dimensional array
    S which has type `np.array[ndim=4,dtype=np.float32]`
    
    length_maps[model_id] is a 1d array with the lengths for the maps
    length_maps[model_id][-1] is the maximum length for the model model_id
    
    example_labeled_start_times[example_id] gives the earliest allowable start
    time the sufficient statistic for starting at time
    t+ example_labeled_start_time[example_id]
    is S[t].
    
    """
    num_examples, example_length, num_features = E.shape
    num_examples, num_models, num_times, num_lengths = L.shape
    
    computeLogLikelihood(E,L,min_lengths,example_labeled_start_times,log_weights,
                         log_M,
                       log_inv_M,max_loglikes_by_example,E_bgd_frame_scores,
                       E_bgd_scores_prefix,E_all_bgd_scores)
    
    # log-likelihood for the data under the current model
    out_ll = 0

    # we compute the normalization factors and the log-likelihood
    # example_lls will be used for normalization in the MStep
    for example_id in range(num_examples):
        # begin by looking at the background log-likelihood for the example
        for length_id in range(num_lengths):
            example_lls[example_id,length_id] = np.exp(E_all_bgd_scores[example_id] - max_loglikes_by_example[example_id])
        
        for model_id in range(num_models):
            for time_id in range(num_times):
                for length_id in range(num_lengths):
                    L[example_id,model_id,
                      time_id,length_id] = np.exp(L[example_id,model_id,
                                                        time_id,length_id] - max_loglikes_by_example[example_id])
                    for length_id2 in range(length_id,num_lengths):
                        example_lls[example_id,length_id2] += L[example_id,model_id,
                                                     time_id,length_id]
                    
                    
        out_ll += np.log(example_lls[example_id]) + max_loglikes_by_example[example_id]

    # # now we update the parameters for the model
    # for example_id in range(num_examples):        
    #     for model_id in range(num_models):
    #         for time_id in range(num_times):
    #             for length_id in range(num_lengths):
    #                 L[example_id,model_id,
    #                   time_id,length_id] = L[example_id,model_id,
    #                                                     time_id,length_id]/example_lls[example_id,0]
    
    return out_ll
            
    
def MStep(E,L,min_lengths,
           example_labeled_start_times,M,
          example_lls
           ):
    """
    We infer the models at this stage
    Algorithm works as follows:
    We run through each example and for each example we run through all
    the models and every time slice that the model has an opinion on
    and then go through all time slices of the example that the model
    time slice could be used on and we compute the weighted average
    using the sufficient statistics computed from the EStep

    example_lls has the normalization factors
    """
    num_examples, num_models, num_times, num_lengths = L.shape
    num_examples, max_time, num_features = E.shape
    for example_id in range(num_examples):        
        for model_id in range(num_models-1):
            example_time_id = 0
            # Each model is defined for times
            # 0 to min_lengths[model_id] + num_lengths
            for model_time_id in range(min_lengths[model_id]):
                # when model_time_id == 0 this means that we are at the start
                # this model frame can only be on example
                # we compute the indices of a window over which we
                # may compute contributions
                # if the max or min are active then we want to know
                # because that is probably an error
                example_window_start_time = max(0,example_labeled_start_times[model_id] - (num_times-1)/2  + model_time_id)                
                # the plus one signifies that this is the frame following the
                # last frame that will be in the window of example frames
                # that can contribute to this model
                example_window_end_time = min(max_time,
                                              example_labeled_start_times[model_id] + (num_times-1)/2 +  model_time_id + 1)
                # for example_frame_id in range(example_window_start_time,
                #                               example_window_end_time):
                for start_time in range(num_times):
                    for length_id in range(num_lengths):
                        for feature_id in range(num_features):
                            M[model_id,model_time_id,feature_id] += L[example_id,model_id,start_time + example_window_start_time,length_id]*E[example_id,start_time+example_window_start_time,feature_id]
                    
            # now we handle the case where the estimate is not in every length version of the model
            for model_time_id in range(min_lengths[model_id],min_lengths[model_id]+num_lengths-1):
                for start_time in range(num_times):
                    for length_id in range(model_time_id-min_lengths[model_id]+1,num_lengths):
                        for feature_id in range(num_features):
                            M[model_id,model_time_id,feature_id] += L[example_id,model_id,start_time + example_window_start_time,length_id]*E[example_id,start_time+example_window_start_time,feature_id]

                if num_lengths > 0:
                    for length_id in range(1,num_lengths):
                        #
                    
                    log_M[model_id,time_id,length_id
                    L[example_id,model_id,
                      time_id,length_id] = L[example_id,model_id,
                                                        time_id,length_id]/example_lls[example_id]
                      
    
