import numpy as np
from scipy.linalg import circulant

#initialize some test data to play with
# in this case shifted sine functions
s=np.sin(np.arange(100)/99. * 10.*np.pi)

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


    
# generate all shifts does signals that are in rows
# we want signals in columns
X = generate_all_shifts(s,20).T

# need to set initial parameters
# X,num_codes,num_sets


signal_length, num_signals = X.shape
num_codes = 10
num_sets = 10


#initialie codebooks
B = normalize_mat(np.random.standard_normal(size=(signal_length,num_codes)))
Phi = normalize_mat(np.random.exponential(size=(num_codes,num_sets)),norm_type=1)

# implementing the split bregman method
# osher etc.

def yin_shrink(y,alpha):
    return np.sign(y) * np.maximum(np.abs(y)-alpha,0)

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


from scipy.linalg import norm

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



u = np.random.standard_normal(num_codes)
u = np.array([ 0.75904652, -0.05970906, -0.48429658, -0.41407721, -0.95183343,
       -0.12282432,  0.64598329, -1.28912995,  0.5389452 , -1.77799504])

u2 = yin_split_bregman(B,X[:,0],1.5,1,.00001,u)
u3 = osher_kicking_bregman(B,X[:,0],1,1,tol=.00001)


u = np.array([ 0.75904652, -0.05970906, -0.48429658, -0.41407721, -0.95183343,
       -0.12282432,  0.64598329, -1.28912995,  0.5389452 , -1.77799504])

from split_bregman import split_bregman
v =np.zeros(num_codes)
u_new = np.zeros(num_codes)
split_bregman(B,u,v,u_new,X[:,0],B.shape[0],B.shape[1],1.1,.1,.00001)

alpha = np.random.exponential(size=(num_sets,))
# need to get a positive definite matrix with all 
# non-negative entries
# implicitly we are thinking of this as a diagonal matrix
# but we are really thinking of this as a diagonal of a 
# diagonal matrix
sigma = np.random.exponential(size=(num_sets,))

lambda_2 = 1
B_tilde = np.vstack((B,np.diag((lambda_2 * np.sum(Phi * np.tile(alpha,(num_codes,1)),axis=1))**(-.5))))
# numpy thinks all arrays are row-vectors
x_tilde = np.hstack((X[:,0],np.zeros(num_sets)))

####
# testing whether that B_tilde method works correctly
#
lambda_2 = 1
np.diag((lambda_2 * np.sum(Phi * np.tile(alpha,(num_codes,1)),axis=1))**(-.5))

phi_sum = np.zeros(np.diag(Phi[:,0]).shape)
for i in xrange(Phi.shape[1]):
    phi_sum+= alpha[i] * np.diag(Phi[:,i])


(lambda_2 * phi_sum)**(-.5)

#
# implementing the set-level optimization
# alternate between alpha and sigma optimization

from sympy import roots,symbols,I
from sympy.utilities.lambdify import lambdify

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


# now we are going to 


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

