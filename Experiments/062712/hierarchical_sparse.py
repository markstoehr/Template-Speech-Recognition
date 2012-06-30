import numpy as np
from scipy.linalg import circulant,norm
from sympy import roots,symbols,I



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


def osher_kicking_bregman(A,f,mu,delta,tol=.00001,
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
    delta: float
        step length
    tol: float
        convergence criterion
    """
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
    
