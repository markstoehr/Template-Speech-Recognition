import numpy as np
import random
#
# Compute EM algorithm
#  uses helper functions
#

class Bernoulli_Mixture:
    def __init__(self,num_mix,data_mat,init_type='unif_rand',
                 opt_type='expected'):
        self.num_mix = num_mix
        self.height,self.width = data_mat.shape[1:]
        self.data_mat = data_mat
        self.num_data = data_mat.shape[0]
        # initializing weights
        self.weights = 1./num_mix * np.ones(num_mix)
        self.opt_type=opt_type
        self.init_affinities_templates(init_type)
        


    def run_EM(self,tol,save_template=False):
        """ EM algorithm
        First we compute the expected value of the label vector for each point
        """
        loglikelihood = -np.inf
        # First E step plus likelihood computation
        new_loglikelihood = self.compute_loglikelihoods()
        while new_loglikelihood - loglikelihood > tol:
            loglikelihood = new_loglikelihood
            # M-step
            self.M_step()
            # E-step
            new_loglikelihood = self.compute_loglikelihoods()
 
    def M_step(self):
        self.weights = np.mean(self.affinities,axis=0)
        for mix_id in xrange(self.num_mix):
            self.templates[mix_id] = np.sum(self.data_mat * np.tile(self.affinities[:,mix_id],
                                                                    (self.width,self.height,1)).transpose(),
                                            axis=0)
            self.templates[mix_id] /= (self.weights[mix_id] * self.num_data)
        self.threshold_templates()
        self.log_templates = np.log(self.templates)
        self.log_invtemplates = np.log(1-self.templates)

        
    def threshold_templates(self):
        self.templates = np.maximum(np.minimum(self.templates,.95),.05)

    def init_affinities_templates(self,init_type):
        if init_type == 'unif_rand':
            random.seed()
            idx = range(self.num_data)
            random.shuffle(idx)
            self.affinities = np.zeros((self.num_data,
                                        self.num_mix))
            self.templates = np.zeros((self.num_mix,
                                       self.height,
                                       self.width))
            for mix_id in xrange(self.num_mix):
                self.affinities[self.num_mix*np.arange(self.num_data/self.num_mix)+mix_id,mix_id] = 1.
                self.templates[mix_id] = np.mean(self.data_mat[self.affinities[:,mix_id]==1],axis=0)
                self.threshold_templates()
        elif init_type == 'specific':
            random.seed()
            idx = range(self.num_data)
            random.shuffle(idx)
            self.affinities = np.zeros((self.num_data,
                                        self.num_mix))
            self.templates = np.zeros((self.num_mix,
                                       self.height,
                                       self.width))
            for mix_id in xrange(self.num_mix):
                self.affinities[self.num_mix*np.arange(self.num_data/self.num_mix)[1]+mix_id,mix_id] = 1.
                self.templates[mix_id] = np.mean(self.data_mat[self.affinities[:,mix_id]==1],axis=0)
                self.threshold_templates()

        self.log_templates = np.log(self.templates)
        self.log_invtemplates = np.log(1-self.templates)


    def get_templates(self):
        return self.templates

    def init_templates(self):
        self.templates = np.zeros((self.num_mix,
                                   self.height,
                                   self.width))

    def get_num_mix(self):
        return self.num_mix

    def get_weights(self):
        return self.weights

    def set_weights(self,new_weights):
        np.testing.assert_approx_equal(np.sum(new_weights),1.)
        assert(new_weights.shape==(self.num_mix,))
        self.weights = new_weights
        

    def compute_loglikelihoods(self):
        template_logscores = self.get_template_loglikelihoods(self.data_mat)
        loglikelihoods = template_logscores + np.tile(np.log(self.weights),(self.num_data,1))
        max_vals = np.amax(loglikelihoods,axis=1)
        # adjust the marginals by a value to avoid numerical
        # problems
        logmarginals_adj = np.sum(np.exp(loglikelihoods - np.tile(max_vals,(self.num_mix,1)).transpose()),axis=1)
        loglikelihood = np.sum(np.log(logmarginals_adj)) + np.sum(max_vals)
        self.affinities = np.exp(loglikelihoods-np.tile(logmarginals_adj+max_vals,
                                           (self.num_mix,1)).transpose())
        self.affinities/=np.tile(np.sum(self.affinities,axis=1),(self.num_mix,1)).transpose()
        return loglikelihood
        
    def get_template_loglikelihood(self,datum):
        return np.sum(self.log_templates[:,datum > .5],\
                           axis=1) +\
                           np.sum(self.log_invtemplates[:,datum < .5],\
                                       axis=1)
    
    def get_template_loglikelihoods(self,data_mat):
        """ Assumed to be called whenever
        """
        return np.array(map(lambda x:\
                                self.get_template_loglikelihood(x),\
                                data_mat))

        
    def set_template_vec_likelihoods(self):
        pass
        

def bernoulli_EM(data_mat,num_mix, tol):
    """ Takes in an iterator that has the data, assumed to be an N by d1 by d2 matrix, where N is the number of examples
    and d1 and d2 are the size of the matrix
    
    Parameters
    ----------
    data_mat: array
    num_mix: int
        Number of mixture centers
    tol: double
        Convergence criterion
    
    Returns
    -------
    mixture_model: Object
    
    
    """
    bm = Bernoulli_Mixture(num_mix,data_mat)
    
    if num_mix == 1:
        bm.set_templates(clean_templates(np.mean(data_mat,axis=0).reshape(1,data_mat.shape[1],data_mat.shape[2])))
        bm.set_weights(np.array([1.]))
    elif num_mix > 1:
        run_EM(bm,data_mat,num_mix,tol)
    return bm

def run_EM(bm,data_mat,num_mix,tol):
    # sample num_mix examples without replacement randomly
    random.seed()
    idx = range(data_mat.shape[0])
    random.shuffle(idx)
    bm.set_templates(data_mat[idx])
    likelihood = -np.inf
    new_likelihood,E_mat = bm.compute_likelihoods(data_mat)
    while new_likelihood - likelihood > tol:
        likelihood = new_likelihood
        # E-step
        bm.set_weights(get_E_weights(E_mat,data_mat))
        # M_step
        bm.update_templates(template_weights,data_mat)
        new_likelihood,E_mat = bm.compute_likelihood(data_mat)

def get_E_weights(E_mat):
    np.tile(np.sum(E_mat,axis=1),(2,1)).transpose()

def E_step(bm,data_mat):
    return bm.template_expectations
    pass
    
def _vec_likelihood_():
    pass
        

def _clean_template(template_estimate,
                    minVal,
                    maxVal):
    return np.maximum(np.minimum(template_estimate,\
                                 .95),\
                      .05)

def _vec_clean_templates(template_estimates,
                        minVal,
                        maxVal):
    vec_clean = np.vectorize(lambda x:\
                                 _clean_template(x,minVal,maxVal))
    return vec_clean(template_estimates)

def clean_templates(template_estimates,
                    minVal = .05,
                    maxVal = .95):
    """ Assumes a 3 dimensional array with the first
        axis being the results
        simply applies a maximum and a minimum criterion
        to the data
    """
    return _vec_clean_templates(template_estimates,minVal,maxVal)
