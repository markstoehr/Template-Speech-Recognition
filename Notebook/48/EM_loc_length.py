#!/usr/bin/python

import numpy as np
from scipy.signal import gaussian
import FastEMLocLength

def computeBackgroundLogLikelihoods(E,log_background_model,log_inv_background_model, object_prior_prob):
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

    log_background_prior = np.log(1-object_prior_prob)

    for feature_id in xrange(num_features):
        background_log_odds[feature_id] = log_background_model[feature_id] - log_inv_background_model[feature_id]

    for example_id in xrange(num_examples):
        for frame_id in xrange(max_example_length):
            E_bgd_frame_scores[example_id,frame_id] = 0
            for feature_id in xrange(num_features):
                E_bgd_frame_scores[example_id,frame_id] += E[example_id,frame_id,feature_id] * background_log_odds[feature_id] + log_inv_background_model[feature_id] + log_background_prior

            if frame_id == 0:
                E_bgd_scores_prefix[example_id,frame_id] = E_bgd_frame_scores[example_id,frame_id]
            else:
                E_bgd_scores_prefix[example_id,frame_id] = E_bgd_frame_scores[example_id,frame_id] + E_bgd_scores_prefix[example_id,frame_id-1]

        E_all_bgd_scores[example_id] = E_bgd_scores_prefix[example_id,max_example_length-1]

    return (E_bgd_frame_scores,
            E_bgd_scores_prefix,
            E_all_bgd_scores)



def initSuffStats(E,num_models,num_times,num_lengths,min_prob=0.01):
    """
    num_times, num_lengths are both assumed to be odd
    """
    num_examples, max_time_length, num_features = E.shape
    L = np.zeros((num_examples,
                  num_models,
                  num_times,
                  num_lengths))
    
    example_lls = np.zeros(num_examples)

    num_latent_values_per_example = num_models*num_times*num_lengths
    example_max_latent_val = 1 - (num_latent_values_per_example-1)*min_prob
    mid_time = (num_times-1)/2
    mid_length = (num_lengths-1)/2
    
    # put a gaussian distribution over the lengths
    length_distributions = gaussian(num_lengths,2)
    length_distributions /= length_distributions.sum()

    # uniform distribution for the times
    start_time_distribution = np.ones(num_times)/float(num_times)

    
    
    L[:] = min_prob

    for i in xrange(num_examples):
        model_idx = i % num_models
        L[i,model_idx] = np.outer(start_time_distribution,length_distributions)
        example_lls[i] = L[i].sum()
        

    return (L, 
            example_lls, 
            np.zeros((num_models,
                      max_time_length,
                      num_features)),
            np.zeros((num_models,                      
                      num_lengths)),
            np.zeros((num_models,
                      num_lengths)))
        

def EM(E,tol,M,min_lengths,example_labeled_start_times,
       num_start_times,num_lengths,background_model,object_prior_prob = .9,
       min_prob=0.01,verbose=True):
    """
    EM algorithm that allows length variation and location variation within the
    labeled examples
    assumed to have a model already initialized
    model has lengths given by min_lengths
    example_labeled_start_times says where the middle start time is for the
       examples.

    M is the model
    """
    
    num_examples = len(E)
    num_models = len(min_lengths)
    min_lengths = min_lengths.astype(np.uint16)
    example_labeled_start_times = example_labeled_start_times.astype(np.uint16)
    
    weights =  np.ones((num_models,num_lengths)) * 1./(num_models*num_lengths)
    log_start_time_priors = np.log( object_prior_prob * np.ones(num_start_times) * (1./num_start_times))
    max_loglikes_by_example = np.zeros(num_examples)
    example_likes = np.zeros(num_examples)
    
    log_background_model = np.log(background_model)
    log_inv_background_model = np.log(1-background_model)

    L = np.zeros((num_examples,num_models,num_start_times,num_lengths),
             dtype=np.float64) 
    LDivideTracker = np.zeros(L.shape,dtype=np.uint8)
    M_normalizations = np.zeros((num_models,num_lengths)).copy()
    


    (E_bgd_frame_scores,
     E_bgd_scores_prefix,
     E_all_bgd_scores) = computeBackgroundLogLikelihoods(E,log_background_model,log_inv_background_model,object_prior_prob)

    criterion = np.inf
    log_like = -np.inf
    
    cur_iter = 0
    while criterion > tol:
        prev_log_like = log_like



        # get the log models
        log_weights = np.log(weights)
        M = np.clip(M,min_prob,1-min_prob)
        log_M = np.log(M)
        log_inv_M = np.log(1-M)
        W = (log_M - log_inv_M).astype(np.float64)
        c = np.zeros((num_models,
                      num_lengths))
        
        for model_id in xrange(num_models):
            c[model_id] = np.cumsum(log_inv_M.sum(-1)[model_id])[min_lengths[model_id]:
                                                                     min_lengths[model_id]+num_lengths]

        L[:] = 0.
        
        L, max_loglikes_by_example = FastEMLocLength.computeObjectFullLogLikelihood(
            E,
            L,
            min_lengths,
            example_labeled_start_times,
            log_weights,
            log_start_time_priors,
            W,
            c,
            max_loglikes_by_example,
            E_bgd_frame_scores,
            E_bgd_scores_prefix,
            E_all_bgd_scores,
            )

        example_likes[:] = 0.

        log_like, L, example_likes = FastEMLocLength.EStep(E,
                                                         L,
                                                         max_loglikes_by_example,
                                                         E_bgd_frame_scores,
                                                         E_bgd_scores_prefix,
                                                         E_all_bgd_scores,
                                                         example_likes,
                                                         np.log(1-object_prior_prob))

        M_normalizations[:] = 0
        
        L,weights = FastEMLocLength.MStep_weights(E,
                                                  L,
                                                  weights,
                                                  example_likes,
                                                  M_normalizations)

        M[:] = 0.
        FastEMLocLength.MStep_models(E,
          L_use,
          min_lengths,
          example_labeled_start_times,
          M2,
          M_normalizations)


        criterion = np.abs(log_like - prev_log_like)/np.abs(log_like)
        cur_iter += 1
        if verbose:
            print "iter %d: %f" % (cur_iter, criterion)

    M = np.clip(M,min_prob,1-min_prob)
    return M, weights, L
        
