import numpy as np

def get_maxima(scores,radius):
    scores_idx = np.tile(np.arange(int(scores.shape[0])).reshape(scores.shape[0],1),(2*radius+1)) + np.arange(2*radius+1)-radius
    scores_idx = scores_idx[radius:-radius]
    return scores >= np.concatenate([np.inf*np.ones(radius),np.max(scores[scores_idx],axis=1),np.inf*np.ones(radius)])
    
