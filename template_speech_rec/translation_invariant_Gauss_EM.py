import numpy as np
from scipy import correlate, convolve
from scipy.linalg import circulant
from _translation_invariant_Gauss_EM import subtract_max_affinities, normalize_affinities

def generate_shifts(atom, num_shifts):
    if num_shifts == 1:
        return np.array([atom])
    else:
        return circulant(np.hstack((atom, np.zeros(num_shifts-1)))).T[:num_shifts][:,num_shifts-1:1-num_shifts]


class TranslationInvariantGaussEM:
    """
    Construct a Gaussian Mixture model
    and add in translation invariance
    We have data x
    the density of the data given the translation and the 
    mixture identity: f(x\mid z,t) is Gaussian
    z _|_ t, z and t are independent
    z is multinomial and t is uniform so the likelihood function
    for the data is
    $$ f(x\mid z,t)f(z)f(t) $$
    
    multinomial model on the mixture identity
    a gaussian model on the data
    """
    def __init__(self,
                 num_mix,
                 data_mat,
                 component_length):
        """
        num_mix is an integer
        data_mat is assumed to be 2-d (so we have a collection
        of 1-d signals).
        component_length is how long the mixture components should be
          this implicitly parameterizes how much translation is allowed
        """
        self.num_mix = num_mix
        self.data_mat = data_mat
        self.num_data, self.data_length = data_mat.shape
        # to make likelihood computation go faster
        self.rep_data_mat = np.tile(self.data_mat.reshape(
                self.num_data,
                1,
                self.data_length),
                                    (1,
                                     self.num_mix,
                                     1))
        assert self.data_mat.ndim == 2
        self.component_length = component_length
        self.trans_amount = self.data_length - self.component_length + 1
        # shifted versions of the data
        self.trans_data_mat = np.array([
                np.array([
                        correlate(datum,unit_vec)
                        for unit_vec in np.eye(self.trans_amount)])
                for datum in self.data_mat])
        self.affinities = np.zeros((self.num_mix,
                                    self.num_data,
                                    self.trans_amount))
        # initialize variables as None so that we know they are
        # defined later
        self.means = None
        self.covs = None
        self.norm_constants = None
        self.mix_weights = None
        self.log_likelihood = - np.inf
        # uniform weighting over the transitions
        self.trans_weights = np.ones(self.trans_amount,dtype=np.float64)
        self.trans_weights /= np.sum(self.trans_weights)
        self.init_templates()
        self.max_affinities = np.zeros(
            self.num_data)

    def init_templates(self):
        self.affinities[:] = 0
        rand_mix_idx = np.random.randint(self.num_mix,
                                         size=(self.num_data))
        self.affinities[rand_mix_idx,
                        np.arange(self.num_data),
                        np.zeros(self.num_data,
                                 dtype=int)] = 1.
        self.mix_weights = np.array([np.sum(rand_mix_idx==i) for i in xrange(self.num_mix)]).astype(np.float32)
        self.mix_weights /= np.sum(self.mix_weights)
       
    def run_EM(self,tol=.0001,num_iters=None):
        self.M_step()
        self.E_step()
        old_log_likelihood = self.log_likelihood/(2.* (1+tol))
        if num_iters == None:
            while (self.log_likelihood - old_log_likelihood)/old_log_likelihood > tol:
                old_log_likelihood = self.log_likelihood
                self.M_step()
                self.E_step()
        else:
            for i in xrange(num_iters):
                if i % 50 == 0:
                    print "On iteration %d" % i
                old_log_likelihood = self.log_likelihood
                self.M_step()
                self.E_step()



    def M_step(self):
        """
        compute the means, sqrt of the diagonal of the covariance matrix
        and we also compute the normalizing constant
        """
        # need to do many convolutions
        # and average over them
        self.marginalized_translations = np.array([
                np.array([ correlate(datum,affinity_row)
                           for datum, affinity_row in zip(self.data_mat,
                                                          component_affinities)])
                for component_affinities in self.affinities])
        self.means = self.marginalized_translations.mean(1)
        print self.means
        self.covs = ((self.marginalized_translations - self.means)**2).mean(1)
        # 
        self.norm_constants = -.5 * np.log((2.*np.pi)**self.num_mix * np.prod(self.covs,1))
        
    def E_step(self):
        """
        Find the likelihoods for each datum for each component and
        each translation

        The likelihood is also computed
        """
        for cur_component, vals in enumerate(zip(self.means,
                                                 self.covs,
                                                 self.norm_constants)):
            mean, cov, norm_constant = vals
            self.affinities[cur_component] = np.sum(- .5 *(self.trans_data_mat - mean)**2 * cov,2) + norm_constant
        self.max_affinities[:] = np.max(
            np.max(self.affinities,
                   0),
            1)
        subtract_max_affinities(self.affinities,self.max_affinities,
                                self.num_mix,
                                self.num_data,
                                self.trans_amount)
        self.affinities = np.exp(self.affinities)
        self.log_likelihood = np.sum(np.log(np.sum(
            np.sum(
                self.affinities * self.trans_weights,
                2).T * self.mix_weights,1)) + self.max_affinities)
        affinity_sums = np.sum(self.affinities.sum(0),1)
        normalize_affinities(self.affinities,affinity_sums,
                             self.num_mix,
                             self.num_data,
                             self.trans_amount)
        self.mix_weights = np.sum(np.sum(self.affinities,1),1)
        self.mix_weights /= np.sum(self.mix_weights)
        
        
        
