import numpy as np
from scipy.linalg import circulant,norm
from sympy import roots,symbols,I


# these are hardcoded to allow for
# faster computation  in get_real_root
# which has an application to the set-coding algorithm
# in the hierarchical sparse coding function
x,a_rootvar,b_rootvar,d_rootvar = symbols('x,a_rootvar,b_rootvar,d_rootvar')
root_list = roots(a_rootvar*x**3 +b_rootvar*x**2 +d_rootvar,x).keys()


root_exprs = [ root_expr.as_real_imag() for root_expr in root_list]


def get_real_root(a,b,d):
    """ Get the real solution
    to the equation in x
    a*x**3 + b*x**2 + d = 0
    """
    real_soln_idx = np.argmin(np.abs([root_exprs[n][1].evalf(subs={a_rootvar:a,b_rootvar:b,d_rootvar:d}) for n in xrange(len(root_exprs))]))
    return root_exprs[real_soln_idx][0].evalf(subs={a_rootvar:a,b_rootvar:b,d_rootvar:d})


def generate_all_shifts(atom, signal_atom_diff):
    """ Shifted version of a matrix with the number
    of shifts being signal_atom_diff
    the atom is a vector and the shifted versions 
    are the rows
    extra zeros are tacked on
    """
    return circulant(np.hstack((atom, np.zeros(signal_atom_diff)))).T[:signal_atom_diff+1]

def normalize_mat(X,norm_type=2):
    """ Normalize by columns
    """
    if norm_type==2:
        return X/np.tile(np.sqrt(np.sum(X*X,axis=0)),(X.shape[0],1))
    elif norm_type==1:
        return X/np.tile(np.sum(np.abs(X),axis=0),(X.shape[0],1))
    else:
        norm_type = float(norm_type)
        return X/np.tile(np.sum(np.abs(X)**norm_type,axis=0)**(1./norm_type),(X.shape[0],1))


def yin_split_bregman(A,f,mu,delta,tol,u=None):
    """ 
    Implement the split bregman method
    Parameters:
    ===========
    A: ndarray ndim=2
        Coding matrix
    u: ndarray ndim=1
        code vector
    f: ndarray ndim=1
        data
    mu: float
        parameter for how strongly to regularize the l1 norm
    delta: float
        step length
    tol: float
        convergence criterion
    """
    signal_length, num_codes = A.shape
    if u is None:
        u = np.random.standard_normal(signal_length)
    # average over the error vectors projected into
    # the code matrix space
    v = np.zeros(num_codes)
    u_diff = tol+1
    u_prev = u.copy()
    while np.sum(np.abs(u_diff)) > tol:
        v= v+ np.dot(A.T, f-np.dot(A,u))
        u=delta*np.sign(v) * np.maximum(np.abs(v)-mu,0)
        u_diff = u-u_prev
        u_prev = u.copy()
        print np.sum(np.abs(u_diff))
    return u


def osher_kicking_bregman(A,f,mu=1,delta=None,tol=.00001,
                          verbose=None):
    """ 
    Implement the linearized kicking bregman
    algorithm
    Parameters:
    ===========
    A: ndarray ndim=2
        Coding matrix
    u: ndarray ndim=1
        code vector
    f: ndarray ndim=1
        data
    mu: float
        parameter for how strongly to regularize the l1 norm
        set to 1 in the bregman paper
    delta: float
        step length
        needs to be set less than \frac{1}{2\|AA^{\top}\|}
        default is None
        if any value is other than default its assumed
        to be less than 1 and greater than zero and we 
        multiply that upper bound by that constant
    tol: float
        convergence criterion
    """
    if delta is None:
        delta = (1./(2.5 * norm(np.dot(A,A.T))))
    else:
        delta *= (1./(2. * norm(np.dot(A,A.T))))
    
    signal_length, num_codes = A.shape
    u = np.zeros(num_codes)
    v = np.zeros(num_codes)
    stop_tol = norm(f) * tol
    do_kick,done_kick=False,False
    num_iter = 0
    Au = np.dot(A,u)
    prev_error = np.inf 
    cur_error = norm(Au - f)
    while prev_error - cur_error > stop_tol or do_kick:
        if verbose > 0:
            if num_iter % verbose == 0:
                print "iteration: %d" % num_iter
                print u
        num_iter += 1
        if do_kick:
            # update direction
            v_up = np.dot(A.T, f-Au)
            # zero indices
            I_0 = np.abs(u) < tol
            # number of steps to take to get out of stagnation
            if np.sum(I_0) > 0:
                s = np.min(np.ceil((mu * np.sign(v_up) - v)/ v_up)[I_0])
                v[I_0]  = v[I_0] + s*v_up[I_0]            
            else:
                v= v+ np.dot(A.T, f-Au)
            do_kick=False
            done_kick=True
        else:            
            v= v+ np.dot(A.T, f-Au)
            done_kick=False
        u_new = delta*np.sign(v) * np.maximum(np.abs(v)-mu,0)
        # <= since we handle the case where we have a zero
        do_kick= norm(u_new - u) <= norm(u_new)*tol
        u = u_new
        Au = np.dot(A,u)
        prev_error = cur_error
        cur_error = norm(Au-f)
        if done_kick and do_kick:
            break
    return u, num_iter

def uzawa_basis_pursuit(Phi,sigma,eta=1,tau=.0445,rho=None,tol=.00001):
    """
    "A Predual Proximal Point Algorithm solving a Non
    Negative Basis Pursuit Denoising Model"
    F. Malgouyres and T. Zeng
    
    Int. J. Computer Vision, 2009

    Parameters:
    ===========
    Phi:
        current dictionary estimate, a matrix
    sigma:
        vector we are trying to approximate
    eta:
        parameter for weighting how much we are stabilizing the
        original (highly unstable optimization problem) heart of the
        uzawa method
        In the experiments reported in the paper
        this controls the sparsity of the approximatino
        larger values give greater sparsity
        smaller values give less sparsity

        This could be increased as the iterations go
        and, in particular, in the paper it was set
        between values of 1 and 100
    tau:
        how much to weight inner product between the predual solution
        .0445 comes from paper, it controls
        how close the sparse approximation should be to
        the actual image, probably the best bet
        is to start at higher values when sparse coding
        is beginning and gradually decrease it
    rho:
       inverse step length 
    tol:
       convergence criterion
    """
    if rho is None:
        rho = uzawa_set_rho(Phi,eta)
    num_codes, num_sets = Phi.shape
    alpha = np.zeros(num_sets)
    alpha_next = np.zeros(num_sets)
    # dual variable we are comparing to the sigma
    u = np.zeros(num_codes)
    u_next = np.zeros(num_codes)
    while norm(u-u_next) >= norm(u_next)*tol:
        u = u_next
        while norm(alpha -alpha_next) >= norm(alpha_next)*tol:
            alpha = alpha_next
            w = 2*eta*u - np.dot(Phi,alpha) + sigma/tau
            w_norm = norm(w)
            if w_norm <= 1:
                w[:] =0
            else:
                w = (w_norm-1)/(2*eta*w_norm) * w
            alpha_next = np.maximum(alpha+rho*(np.dot(w,Phi)-1),0)
        u_next = w
    return alpha


def solve_sigma_set_coding(lambda_2,W,lambda_3,Phi,alpha):
    """
    Use polynomials loaded above to solve for the sigma
    which is the diagonal of the diagonal matrix estimate
    for the covariance matrix of the codes, W

    solve the polynomial

    \min \frac{\lambda_2}{n}\sum_i w_{i,j}^2 \sigma_j^{-1}
       + \lambda_3 (\sigma_j 
          - \Phi_{j,1:num_signals} \alpha)^2
         
    -\frac{\lambda_2}{n}\sum_i w_{i,j}^2 +
          2\lambda_3 \sigma^2 
       - 2\lambda_3 \Phi_{j,1:num_signals} \alpha \sigma^3
       = 0
    """
    num_codes,num_signals = W.shape
    num_signals = float(num_signals)
    # sigma is the diagonal
    sigma = np.zeros(num_codes)
    w_sq_sums = np.sum(W * W,axis=1)
    Phi_alpha_prods = np.dot(Phi,alpha)
    for i in xrange(num_codes):
        sigma[i] = get_real_root(2* lambda_3,
                                 - 2.*lambda_3 *Phi_alpha_prods[i],
                                 -lambda_2/num_signals * w_sq_sums[i])
    return sigma
        


def hierarchical_sparse_coding_iteration(X,B,Phi,alpha,
                               lambda_1,lambda_2,
                               lambda_3,lambda_4=None,
                               tol=.00001,
                               delta=None,
                               verbose=False):
    """
    Sparse coding portion of the dictionary learning algorithm relies
    on several of the optimization procedures considered above

    An iterate that first optimizes W with alpha fixed and then optimizes alpha with W fixed

    First part of the algorithm gets the representation
    in terms of the data using the osher_kicking_bregman
    algorithm

    The next part uses the predual uzawa nonnegative basis pursuit
    algorithm to finish off.

    Parameters:
    ==========
    X:
       Data to code
    B: 
       Codebook for the data, this can be thought of as
       a dictionary meant to capture the covariance structure
       of the data, namely, expressed in this basis
       the data should have a diagonal covariance structure
       recalling back the macrotile algorithm
       considered by Donoho and Mallat
    Phi:
       Set coding dictionary, this is a dictionary that
       we can use  to sparsely represent the codes
    lambda_1:
       Tradeoff between sparsity and decomposition in the
       first sparse coding problem
    lambda_3:
       From the hierarchical sparse coding paper
       this should be set to be large as it controls
       the sparsity of the  coefficients
    lambda_4:
       This corresponds to the tradeoff
       between sparsity of A and not
    """
    signal_length, num_signals = X.shape
    num_codes = B.shape[1]
    num_sets = Phi.shape[1]
    # transform dictionary so that we get the modified
    # elastic net problem into a lasso problem
    B_tilde = np.vstack((B,np.diag((lambda_2 * np.sum(Phi * np.tile(alpha,(num_codes,1)),axis=1))**(-.5))))
    X_tilde = np.vstack((X,np.zeros((num_codes,num_signals))))
    # matrix of codes
    W = np.zeros((num_codes,num_signals))
    # loop over every datum/patch/signal window x
    for i in xrange(num_signals):
        W[:,i], _ = osher_kicking_bregman(B_tilde,X_tilde[:,i],mu=lambda_1,
                                     delta=delta,tol=tol,
                                     verbose=verbose)
    # we now perform an alternating minimization
    # between 
    # initialize by the diagonal of the sample
    # covariance of the vectors
    not_converged = True
    prev_alpha = alpha
    prev_sigma = np.sum(W*W,axis=1)
    while not_converged:
        # eta = lambda_4 since eta makes the solution sparser
        # as we increase it
        alpha = uzawa_basis_pursuit(Phi,sigma,eta=lambda_4,tau=.0445,rho=None,tol=.00001)
        sigma = solve_sigma_set_coding(lambda_2,W,lambda_3,Phi,alpha)
        not_converged = (norm(alpha-prev_alpha) >= norm(alpha)*tol) or\
            (norm(sigma-prev_sigma) >= norm(sigma)*tol)
        prev_alpha = alpha
        prev_sigma = sigma
    return W,alpha
    
    
def hierarchical_sparse_coding(X,B,Phi,alpha,
                               lambda_1,lambda_2,
                               lambda_3,lambda_4=None,
                               tol=.00001,
                               delta=None,
                               verbose=False):
    """
    Main loop for the sparse coding, runs throughv various
    iterations, the problem is convex so it converges
    to a global optimum, however there are many parameters
    to set that mean we could get to the optimum slowly or 
    not at all!
    """
    prev_alpha = np.zeros(Phi.shape[1])
    prev_W = np.zeros((B.shape[1],X.shape[1]))
    not_converged = True
    while not_converged:
        W,alpha = hierarchical_sparse_coding_iteration(X,B,Phi,alpha,
                                                       lambda_1,lambda_2,
                                                       lambda_3,lambda_4=None,
                                                       tol=.00001,
                                                       delta=None,
                                                       verbose=verbose)
        not_converged = (norm(alpha-prev_alpha) >= norm(alpha)*tol) or\
            (norm(W-prev_W) >= norm(W)*tol)
        prev_alpha = alpha
        prev_W = sigma
    return W,alpha


    
def hierarchical_sparse(X,num_codes,num_sets):
    """
    Following optimization is performed:
    $$(\hat{W}, \hat{\alpha}) = 
         \underset{W,\alpha}{\arg\min} L(W,\alpha) 
         + \frac{\lambda_1}{n}\|W\|_1 +\gamma\|\alpha\|_1$$
    subject to $\alpha\succ 0$

    The loss function $L(W,\alpha)$ being minimized is

    $$ \frac{1}{n} \sum_{i=1}^n \left{\frac{1}{2}\|x_i-Bw_I\|^2 
                 +\lambda_2 w_i^\top \Omega(\alpha)w_i \right}$$

    Here $W = (w_1,\ldots,w_n) \in \mathbb{R}^{num_codes\times num_signals}$ are
    the first level (patch-level) descriptions,

    $\alpha\in\mathbb{R}^{num_sets}$ is the set-level representation and
    the penalty term is

    $\Omega(\alpha) \equiv \left( \sum_{k=1}^{num_sets} \alpha_k \text{diag}(\phi_k)\right)^{-1}

    We solve this using an alternating minimization procedure

    Parameters:
    ===========
    X: ndarry
        Columns are windows of the signal, rows separate the diferent windows
        signal_length by num_signals numpy array
    num_codes: int
        Number of columns in the code matrix and number of rows in the set (second-level) code matrix
    num_sets: int
        Number of columsn in the set (second-level) code matrix
    

    Return Parameters:
    =================
    B: ndarray
        Patch level code matrix for the patches
        signal_length by num_codes
    Phi: ndarray
        second-level set-level dictionary
        num_codes by 
    W: ndarray
        patch-level description in terms of the codes 
        in the B matrix
    """
    pass
    
