import amitgroup as ag
import numpy as np
import random, collections


BernoulliMixtureSimple = collections.namedtuple('BernoulliMixtureSimple',
                                                'log_templates log_invtemplates weights')

class BernoulliMixture:
    """
    Bernoulli Mixture model with an EM solver.

    Parameters
    ----------
    num_mix : int
        Number of mixture components. 
    data_mat : ndarray
        Binary data array. Can be of any shape, as long as the first axis separates data entries.
        Values in the data must be either 0 or 1 for the algorithm to work.
    init_type : string
        Specifies the algorithm initialization.
         * `unif_rand` : Unified random. 
         * `specific` : TODO: Added explanation of this.
    
    Attributes 
    ----------
    num_mix : int
        Number of mixture components.
    num_data : int
        Number of data entries.
    data_length : int
        Length of data flattened. 
    iterations : int
        Number of iterations from the EM. Will be set after calling :py:func:`run_EM`.      
    templates : ndarray 
        Mixture components templates. Array of shape ``(num_mix, A, B, ...)``, where ``A, B, ...`` is the shape of a data entry.
    work_templates : ndarray
        Flattened mixture components templates. Array of shape ``(num_mix, data_length)``.
    weights : ndarray
        The probabilities of drawing from a certain mixture component. Array of length ``num_mix``.
    affinities : ndarray
        The contribution of each original data point to each mixture component. Array of shape ``(num_data, num_mix)``.

    Examples
    --------
    Create a mixture model with 2 mixture componenents.
    
    >>> import amitgroup as ag
    >>> import numpy as np
    >>> data = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]]) 
    >>> mixture = ag.stats.BernoulliMixture(2, data)
    
    Run the algorithm until specified tolerance.
    
    >>> mixture.run_EM(1e-3)

    Display the mixture templates and the corresponding weights.
            
    >>> mixture.templates
    array([[ 0.95      ,  0.95      ,  0.50010438],
           [ 0.05      ,  0.05      ,  0.95      ]])
    >>> mixture.weights
    array([ 0.66671347,  0.33328653])

    Display the latent variable, describing what combination of mixture components
    a certain data frame came from:
    
    >>> mixture.affinities
    array([[  9.99861515e-01,   1.38484719e-04],
           [  2.90861524e-03,   9.97091385e-01],
           [  9.97376426e-01,   2.62357439e-03]])

    """
    def __init__(self,num_mix,data_mat,init_type='unif_rand',init_seed=0):
        # TODO: opt_type='expected'
        self.num_mix = num_mix
        self.num_data = data_mat.shape[0]
        self.data_shape = data_mat.shape[1:]
        # flatten data to just be binary vectors
        self.data_length = np.prod(data_mat.shape[1:])
        self.data_mat = data_mat.reshape(self.num_data, self.data_length)
        self.iterations = 0
        # set the random seed
        self.seed = init_seed
        np.random.seed(self.seed)


        self.min_probability = 0.05 

        # If we change this to a true bitmask, we should do ~data_mat
        self.not_data_mat = 1 - self.data_mat
        
        # initializing weights
        self.weights = 1./num_mix * np.ones(num_mix)
        #self.opt_type=opt_type TODO: Not used yet.
        self.init_affinities_templates(init_type)

        # Data sizes:
        # data_mat : num_data * data_length
        # weights : num_mix
        # work_templates : num_mix * data_length
        # affinities : num_data * num_mix


    # TODO: save_template never used!
    def run_EM(self, tol, min_probability=0.05, debug_plot=False):
        """ 
        Run the EM algorithm to specified convergence.
        
        Parameters
        ----------
        tol : float
            The tolerance gives the stopping condition for convergence. 
            If the loglikelihood decreased with less than ``tol``, then it will break the loop.
        min_probability : float
            Disallow probabilities to fall below this value, and extend below one minus this value.
        init_seed : integer or None
        """
        self.min_probability = min_probability 
        loglikelihood = -np.inf
        # First E step plus likelihood computation
        new_loglikelihood = self._compute_loglikelihoods()

        if debug_plot:
            plw = ag.plot.PlottingWindow(subplots=(1, self.num_mix), figsize=(self.num_mix*3, 3))

        self.iterations = 0
        while (new_loglikelihood - loglikelihood)/loglikelihood > tol:
            ag.info("Iteration {0}: loglikelihood {1}".format(self.iterations, loglikelihood))
            loglikelihood = new_loglikelihood
            # M-step
            self.M_step()
            # E-step
            new_loglikelihood = self._compute_loglikelihoods()
            
            self.iterations += 1

            if debug_plot and not self._plot(plw):
                raise ag.AbortException 

        self.set_templates()
        

    def _plot(self, plw):
        if not plw.tick():
            return False 
        self.set_templates()
        for m in xrange(self.num_mix):
            # TODO: Fix this somehow
            if self.templates.ndim == 3:
                plw.imshow(self.templates[m], subplot=m)
            elif self.templates.ndim == 4:
                plw.imshow(self.templates[m].mean(axis=0), subplot=m)
            else:
                raise ValueError("debug_plot not supported for 5 or more dimensional data")
        return True
 
    def M_step(self):
        self.weights = np.mean(self.affinities,axis=0)
        import pdb; pdb.set_trace()
        self.work_templates = np.dot(self.affinities.T, self.data_mat)
        self.work_templates /= self.num_data 
        self.work_templates /= self.weights.reshape((self.num_mix, 1))
        self.threshold_templates()
        self.log_templates = np.log(self.work_templates)
        self.log_invtemplates = np.log(1-self.work_templates)

    def get_bernoulli_mixture_named_tuple():
        return BernoulliMixtureSimple(log_templates=self.log_templates,
                                      log_invtemplates=self.log_invtemplates,
                                      weights=self.weights)
                                      
        
    def threshold_templates(self):
        self.work_templates = np.clip(self.work_templates, self.min_probability, 1-self.min_probability) 

    def init_affinities_templates(self,init_type):
        if init_type == 'unif_rand':
            random.seed()
            idx = range(self.num_data)
            random.shuffle(idx)
            self.affinities = np.zeros((self.num_data,
                                        self.num_mix))
            self.work_templates = np.zeros((self.num_mix,
                                       self.data_length))
            for mix_id in xrange(self.num_mix):
                self.affinities[self.num_mix*np.arange(self.num_data/self.num_mix)+mix_id,mix_id] = 1.
                self.work_templates[mix_id] = np.mean(self.data_mat[self.affinities[:,mix_id]==1],axis=0)
                self.threshold_templates()
        elif init_type == 'specific':
            random.seed()
            idx = range(self.num_data)
            random.shuffle(idx)
            self.affinities = np.zeros((self.num_data,
                                        self.num_mix))
            self.work_templates = np.zeros((self.num_mix,
                                       self.data_length))
            for mix_id in xrange(self.num_mix):
                self.affinities[self.num_mix*np.arange(self.num_data/self.num_mix)[1]+mix_id,mix_id] = 1.
                self.work_templates[mix_id] = np.mean(self.data_mat[self.affinities[:,mix_id]==1],axis=0)
                self.threshold_templates()

        self.log_templates = np.log(self.work_templates)
        self.log_invtemplates = np.log(1-self.work_templates)

    def init_templates(self):
        self.work_templates = np.zeros((self.num_mix,
                                   self.data_length))
        self.templates = np.zeros((self.num_mix,
                                   self.data_length))

    def set_templates(self):
        self.templates = self.work_templates.reshape((self.num_mix,)+self.data_shape)
        self.log_templates = np.log(self.templates)
        self.log_invtemplates = np.log(1-self.templates)
                                     

    def set_weights(self,new_weights):
        np.testing.assert_approx_equal(np.sum(new_weights),1.)
        assert(new_weights.shape==(self.num_mix,))
        self.weights = new_weights
        
    def _compute_loglikelihoods(self):
        template_logscores = self.get_template_loglikelihoods()

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

    def compute_loglikelihood(self,datamat):
        #num_data = datamat.shape[0]
        #np.tile(self.log_templates 
        pass

    def get_template_loglikelihoods(self):
        """ Assumed to be called whenever
        """
        return np.dot(self.data_mat, self.log_templates.T) + \
               np.dot(self.not_data_mat, self.log_invtemplates.T)
      
    def set_template_vec_likelihoods(self):
        pass
    
    def save(self, filename, save_affinities=False):
        """
        Save mixture components to a numpy npz file.
        
        Parameters
        ----------
        filename : str
            Path to filename
        save_affinities : bool
            Save ``affinities`` or not. This is an option since this will proportional to input data size, which
            can be much larger than simply the mixture templates. 
        """
        entries = dict(templates=self.templates, weights=self.weights)
        if save_affinities:
            entries['affinities'] = self.affinities
        np.savez(filename, **entries) 


def compute_likelihood(bernoulli_mixture,
                       data_mat,ignore_weights=True):
    """
    Compute the likelihood of the model on the data. Should work with either 
    a named tuple mixture representation or a BernoulliMixture object
    """
    num_data = data_mat.shape[0]
    affinities = np.array([ np.tile(log_template,
            (num_data,1)) * data_mat + np.tile(log_invtemplate,
                                               (num_data,1)) *(
                    1 - data_mat) for log_template, log_invtemplate in zip(
                    bernoulli_mixture.log_templates,
                    bernoulli_mixture.log_invtemplates)])
    
