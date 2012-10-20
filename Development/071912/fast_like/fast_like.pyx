# cython fast_like.pyx
# $ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -lgslcblas -lm -o fast_like.so fast_like.c
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np

DTYPE = np.float64
DTYPE = np.uint8

cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline int int_max(int a, int b): return a if a > b else b

cdef extern from "math.h":
    cdef extern float exp(float x)
    cdef extern double log(double x)
    cdef extern float sqrt(float x)
    cdef extern float pow(float x, float y)


cdef extern from "cblas.h":
    float ddot "cblas_sdot"(int N,
                            float *X, int incX,
                            float *Y, int incY)

    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans
    enum CBLAS_UPLO:      CblasUpper, CblasLower
    enum CBLAS_DIAG:      CblasNonUnit, CblasUnit
    enum CBLAS_SIDE:      CblasLeft, CblasRight
    
    void lib_sgemm "cblas_sgemm"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 CBLAS_TRANSPOSE TransB, int M, int N, int K,
                                 float  alpha, float  *A, int lda, float  *B, int ldb,
                                 float  beta, float  *C, int ldc)

    void lib_dgemv "cblas_dgemv"(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                                 int M, int N, double  alpha, double  *A, int lda,
                                               double  *x, int dx,
                                 double  beta,  double  *y, int dy)




ctypedef np.float64_t dtype_t64
ctypedef np.float64_t dtype_t
ctypedef np.uint8_t dtype_bool_t

cdef void sgemm(np.ndarray[dtype_t, ndim=2] A,
                np.ndarray[dtype_t, ndim=2] B,
                np.ndarray[dtype_t, ndim=2] C):
    lib_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans, C.shape[0], C.shape[1],
              B.shape[0], 1.0, <float*>A.data, A.shape[1], <float*>B.data,
              B.shape[1], 0.0, <float*>C.data, C.shape[1])

# transpose version
cdef void dgemv_T(dtype_t* W,
                   dtype_t* bg,
                   dtype_t* W_bg,
                   int W_height,
                   int W_length):
    lib_dgemv(CblasRowMajor, CblasTrans, W_height,
              W_length, 1.0, W, W_length, bg,
              1, 0.0, W_bg,1)

cdef void dgemv(dtype_t* W,
                   dtype_t* bg,
                   dtype_t* W_bg,
                   int W_height,
                   int W_length):
    lib_dgemv(CblasRowMajor, CblasNoTrans, W_height,
              W_length, 1.0, W, W_length, bg,
              1, 0.0, W_bg,1)

def test_dgemv_T(np.ndarray[dtype_t,ndim=1] W,
                  np.ndarray[dtype_t,ndim=1] bgd,
                  np.ndarray[dtype_t,ndim=1] W_bgd,
                  int W_height,
                  int W_length):
    dgemv_T(<dtype_t*> W.data,
                   <dtype_t*>bgd.data,
                   <dtype_t*>W_bgd.data,
                   W_height,
                   W_length)


cdef dtype_t compute_W(dtype_t* T,
                       dtype_t* bg,
                       dtype_t* C_inv_long,
                       dtype_t* W,
                       int T_num_rows,
                       int T_num_cols):
    cdef Py_ssize_t i,j
    cdef dtype_t C = 0.0
    cdef dtype_t cur_bg_val = 0.0
    for i in range(T_num_rows):
         cur_bg_val = bg[i]
         for j in range(T_num_cols):
             C_inv_long[i*T_num_cols + j]= log((1.-T[i*T_num_cols+j])/(1-cur_bg_val))
             C += C_inv_long[i*T_num_cols+ j]
             W[i*T_num_cols + j] = log((T[i*T_num_cols+j])/(cur_bg_val)) - C_inv_long[i*T_num_cols + j]
    return C
             
 

# def fast_like(np.ndarray[dtype_t, ndim=2] T,
#               int T_num_rows,
#               int T_num_cols,
#               np.ndarray[dtype_bool_t,ndim=2] E_window,
#               int E_num_cols,
#               int num_detections,
#               int pad_front,
#               np.ndarray[dtype_bool_t,ndim=2] pad_matrix,
#               np.ndarray[dtype_t,ndim=2] W_bg,
#               np.ndarray[dtype_t,ndim=1] detect_sums,
#               np.ndarray[dtype_t, ndim=1] bg,
#               np.ndarray[dtype_t, ndim=2] C_inv_long,
#               np.ndarray[dtype_t,ndim=2] W):

cdef void compute_padding_scores(dtype_t* W_bg,
                                 dtype_t* detect_sums,
                             int num_detections,
                             int T_num_cols,
                             int pad_front,
                             int E_num_cols):
    """
    Uses the parameter of how much front padding there is and also
    computes how much back padding there should be the back padding is
    computed by looking at whether the end of the template or filter
    as given by T_num_cols will be beyond the number of columns in the
    E_window (as given by E_num_cols) after we do our detections
    (which number to num_detections) given that our starting point is
    at index -pad_front (accounting for the front padding then)

    The somewhat complicated looking loops to compute
    the padding sums just relates to the fact that 
    """
    cdef Py_ssize_t t,t2
    cdef int back_pad_end = (T_num_cols 
                             + num_detections-1 
                             - E_num_cols 
                             - pad_front)
    cdef dtype_t cur_val = 0.0
    for t in range(num_detections):
        detect_sums[t] = 0.0
    if pad_front >0:
        for t in range(pad_front):
            cur_val = W_bg[t]
            for t2 in range(int_min(pad_front-t,num_detections)):
                detect_sums[t2] += cur_val
    if back_pad_end >0:
        for t in range(back_pad_end):
            # first cur_val is the last W_bg value
            cur_val = W_bg[T_num_cols-t]
            for t2 in range(int_min(back_pad_end-t,num_detections)):
                # when t is zero
                # this goes over the entirety of the 
                # back-padding
                detect_sums[num_detections-1-t2] += cur_val


cdef void perform_convolution(dtype_t* detect_sums,
                              dtype_bool_t* E_row,
                              dtype_t* W_row,
                              int pad_front,
                              int W_num_cols,
                              int E_num_cols,
                              int num_detections):
    """
    """
    # figure out what the mapping between the hypothetical
    # convolution output vector and the detect_sums vector
    # depends on pad_front
    # if pad_front = 0, the detect_sums[0] corresponds to
    # out_convolve[W_num_cols-1]
    # out_convolve[k] = E_row[k] * W_row[W_num_cols-1]
    #                 + E_row[k-1] * W_row[W_num_cols-2]
    #                 + ....
    #                 + E_row[k-W_num_cols+1] * W_row[0]
    #
    # if pad_front = 1 then detect_sums[0] corresponds to
    # out_convolve[W_num_cols-2]
    # if pad_front = W_num_cols then we start with detect_sums[1]
    # namely we start with detect_sums[k] where
    # k=int_max(0, pad_front -W_num_cols + 1)
    #
    # the next question is to which entry of out_convolve
    # does detect_sums[k] correspond?
    # in the case where k > 0 , detect_sums[k] correponds to
    # out_convolve[0] = E_row[0] * W_row[W_num_cols-1]
    # if k=0 then it is out_convolve[W_num_cols-1-pad_front]
    
    # position in num_detections that we begin in
    cdef Py_ssize_t start_detect_idx = int_max(0, pad_front- W_num_cols+1)
    # check to make sure that we have some detections to do
    if start_detect_idx >= num_detections:
        return
    # position in out_convolve that we begin in
    cdef Py_ssize_t end_E_idx = int_min(E_num_cols-1,
                                        W_num_cols-1 
                                        + num_detections-1
                                        - start_detect_idx 
                                        - pad_front)
    cdef Py_ssize_t detect_max_idx,detect_min_idx, detect_idx, E_idx
    for E_idx in range(0,end_E_idx+1):
        if E_row[E_idx] > 0:
            max_detect_idx = int_min(E_idx+pad_front,
                                     num_detections-1)
            min_detect_idx = int_max(0,E_idx-W_num_cols+1
                                     +pad_front)
            for detect_idx in range(min_detect_idx,
                                    max_detect_idx+1):
                detect_sums[detect_idx] += W_row[E_idx-detect_idx+pad_front]
            

def test_convolution(np.ndarray[dtype_bool_t, ndim=1] E_row,
                     np.ndarray[dtype_t, ndim=1] W_row,
                     np.ndarray[dtype_t, ndim=1] detect_sums,
                     int E_num_cols,
                     int W_num_cols,
                     int num_detections,
                     int pad_front):
    perform_convolution(<dtype_t*> detect_sums.data,
                              <dtype_bool_t*> E_row.data,
                              <dtype_t*> W_row.data,
                              pad_front,
                              W_num_cols,
                              E_num_cols,
                              num_detections)


cdef void filter_data(dtype_t* W,
                       dtype_t* detect_sums,
                       dtype_bool_t* E_window,
                       int num_detections,
                       int T_num_cols,
                       int T_num_rows,
                       int pad_front,
                       int E_num_cols):
    """
    Computing the sums
    """
    cdef int back_pad_end = (T_num_cols 
                             + num_detections-1 
                             - E_num_cols 
                             - pad_front)
    # each feature row is different
    # feature rows correspond frequency bands
    #   and edge
    # f is for the feature index
    # n is for the current detection time
    # t is the time
    cdef Py_ssize_t f,n,t,e
    for f in range(T_num_rows):
        perform_convolution(detect_sums,
                             &E_window[f*E_num_cols],
                             &W[f*T_num_cols],
                             pad_front,
                             T_num_cols,
                             E_num_cols,
                             num_detections)
                

cdef dtype_t _fast_like2_handmult(np.ndarray[dtype_t, ndim=2] T,
              np.ndarray[dtype_t, ndim=1] bg,
              np.ndarray[dtype_t, ndim=2] C_inv_long,
              np.ndarray[dtype_t, ndim=2] W,
              int T_num_rows,
              int T_num_cols,
              np.ndarray[dtype_bool_t,ndim=2] E_window,
              int E_num_cols,
              int num_detections,
              int pad_front,
              np.ndarray[dtype_t,ndim=1] W_bg,
              np.ndarray[dtype_t,ndim=1] detect_sums):
    """
    
    num_detections:
        If this is greater than E_num_cols then we do padding, this is the number of places that we begin
    pad_front:
        how many background vectors to put at front,
        also says how far to start from the beginning,
        we use this to figure out the amount of back padding
        If this is zero then we start all the detections
        from the front of the vector        
    W_bg:
        product vector between the filter W and the
        background, this allows us to compute
        the contribution to the filter sums that comes
        from the padding
    """
    # assumed that this is the height for E_window and the length of bgd
    cdef dtype_t C = compute_W(<dtype_t*> T.data,
                       <dtype_t*> bg.data,
                       <dtype_t*> C_inv_long.data,
                       <dtype_t*> W.data,
                       T_num_rows,
                       T_num_cols)
    # compute the background and W matchup
    cdef Py_ssize_t i,j
    for i in range(T_num_cols):
        for j in range(T_num_rows):
            W_bg[i] += W[j*T_num_cols + i] * bg[j]
    # computed the pad-matrix
    # f is the frequency/edge type index
    # t is the time, the current place in the
    # num_detections
    # also does the preliminary computation for getting
    # the padding sums
    compute_padding_scores(<dtype_t*> W_bg.data,
                       <dtype_t*> detect_sums.data,
                       num_detections,
                       T_num_cols,
                       pad_front,
                       E_num_cols)
    # we now get the detect sums
    # now we do the sums based on binary features
    filter_data(<dtype_t*> W.data,
                 <dtype_t*> detect_sums.data,
                 <dtype_bool_t*> E_window.data,
                 num_detections,
                 T_num_cols,
                 T_num_rows,
                 pad_front,
                 E_num_cols)
    return C


cdef dtype_t _fast_like2(np.ndarray[dtype_t, ndim=2] T,
              np.ndarray[dtype_t, ndim=1] bg,
              np.ndarray[dtype_t, ndim=2] C_inv_long,
              np.ndarray[dtype_t, ndim=2] W,
              int T_num_rows,
              int T_num_cols,
              np.ndarray[dtype_bool_t,ndim=2] E_window,
              int E_num_cols,
              int num_detections,
              int pad_front,
              np.ndarray[dtype_t,ndim=1] W_bg,
              np.ndarray[dtype_t,ndim=1] detect_sums):
    """
    
    num_detections:
        If this is greater than E_num_cols then we do padding, this is the number of places that we begin
    pad_front:
        how many background vectors to put at front,
        also says how far to start from the beginning,
        we use this to figure out the amount of back padding
        If this is zero then we start all the detections
        from the front of the vector        
    W_bg:
        product vector between the filter W and the
        background, this allows us to compute
        the contribution to the filter sums that comes
        from the padding
    """
    # assumed that this is the height for E_window and the length of bgd
    cdef dtype_t C = compute_W(<dtype_t*> T.data,
                       <dtype_t*> bg.data,
                       <dtype_t*> C_inv_long.data,
                       <dtype_t*> W.data,
                       T_num_rows,
                       T_num_cols)
    # compute the background and W matchup
    dgemv_T(<dtype_t*> W.data,
             <dtype_t*> bg.data,
             <dtype_t*> W_bg.data,
             T_num_rows,
             T_num_cols)
    # computed the pad-matrix
    # f is the frequency/edge type index
    # t is the time, the current place in the
    # num_detections
    # also does the preliminary computation for getting
    # the padding sums
    compute_padding_scores(<dtype_t*> W_bg.data,
                       <dtype_t*> detect_sums.data,
                       num_detections,
                       T_num_cols,
                       pad_front,
                       E_num_cols)
    # we now get the detect sums
    # now we do the sums based on binary features
    filter_data(<dtype_t*> W.data,
                 <dtype_t*> detect_sums.data,
                 <dtype_bool_t*> E_window.data,
                 num_detections,
                 T_num_cols,
                 T_num_rows,
                 pad_front,
                 E_num_cols)
    return C



cdef dtype_t _fast_like_single(dtype_t* T,
                       dtype_t* bg,
                       dtype_t* C_inv_long,
                       dtype_t* W,
                       int T_num_rows,
                       int T_num_cols,
                       np.ndarray[dtype_bool_t,ndim=2] E_window,
                       int E_num_cols,
                       int num_detections,
                       int pad_front,
                       dtype_t* W_bg,
                       np.ndarray[dtype_t,ndim=1] detect_sums):
    """
    
    num_detections:
        If this is greater than E_num_cols then we do padding, this is the number of places that we begin
    pad_front:
        how many background vectors to put at front,
        also says how far to start from the beginning,
        we use this to figure out the amount of back padding
        If this is zero then we start all the detections
        from the front of the vector        
    W_bg:
        product vector between the filter W and the
        background, this allows us to compute
        the contribution to the filter sums that comes
        from the padding
    """
    # assumed that this is the height for E_window and the length of bgd
    cdef dtype_t C = compute_W(T,
                               bg,
                               C_inv_long,
                               W,
                               T_num_rows,
                               T_num_cols)
    # compute the background and W matchup
    cdef Py_ssize_t i,j
    for i in range(T_num_cols):
        for j in range(T_num_rows):
            W_bg[i] += W[j*T_num_cols + i] * bg[j]
    # computed the pad-matrix
    # f is the frequency/edge type index
    # t is the time, the current place in the
    # num_detections
    # also does the preliminary computation for getting
    # the padding sums
    compute_padding_scores(W_bg,
                       <dtype_t *>detect_sums.data,
                       num_detections,
                       T_num_cols,
                       pad_front,
                       E_num_cols)
    # we now get the detect sums
    # now we do the sums based on binary features
    filter_data(W,
                 <dtype_t *> detect_sums.data,
                 <dtype_bool_t *> E_window.data,
                 num_detections,
                 T_num_cols,
                 T_num_rows,
                 pad_front,
                 E_num_cols)
    return C


def templates_examples_fast_like(np.ndarray[dtype_t, ndim=3] Ts,
                                          np.ndarray[dtype_t, ndim=2] bgs,
                                          np.ndarray[dtype_bool_t,ndim=3] E_windows,
                    np.ndarray[int,ndim=1] E_num_cols_array,
                    int num_detections,
                    int pad_front):
    cdef int num_T = Ts.shape[0]
    cdef int T_num_rows = Ts.shape[1]
    cdef int T_num_cols = Ts.shape[2]
    cdef int num_E_examples = E_windows.shape[0]
    Cs = np.zeros((num_T,num_E_examples),dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim=1] W_bg = np.zeros(T_num_cols)
    cdef np.ndarray[dtype_t,ndim=2] W = np.zeros((T_num_rows,
                                                  T_num_cols))
    cdef np.ndarray[dtype_t,ndim=2] C_inv_long = np.zeros((T_num_rows,
                                                          T_num_cols))
    detect_sums = np.zeros((num_T,
                                                            num_E_examples,
                                                            num_detections),dtype=np.float64)
    cdef Py_ssize_t cur_E_idx, cur_T_idx
    for cur_T_idx in range(num_T):
        for cur_E_idx in range(num_E_examples):
            Cs[cur_T_idx,cur_E_idx] = _fast_like2(Ts[cur_T_idx],
                                        bgs[cur_E_idx],
                                        C_inv_long,
                                        W,
                                        T_num_rows,
                                        T_num_cols,
                                        E_windows[cur_E_idx],
                                        E_num_cols_array[cur_E_idx],
                                        num_detections,
                                        pad_front,
                                        W_bg,
                                        detect_sums[cur_T_idx,cur_E_idx])
    return Cs,detect_sums

def te_fl_handmult(np.ndarray[dtype_t, ndim=3] Ts,
                                          np.ndarray[dtype_t, ndim=2] bgs,
                                          np.ndarray[dtype_bool_t,ndim=3] E_windows,
                    np.ndarray[int,ndim=1] E_num_cols_array,
                    int num_detections,
                    int pad_front):
    cdef int num_T = Ts.shape[0]
    cdef int T_num_rows = Ts.shape[1]
    cdef int T_num_cols = Ts.shape[2]
    cdef int num_E_examples = E_windows.shape[0]
    Cs = np.zeros((num_T,num_E_examples),dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim=1] W_bg = np.zeros(T_num_cols)
    cdef np.ndarray[dtype_t,ndim=2] W = np.zeros((T_num_rows,
                                                  T_num_cols))
    cdef np.ndarray[dtype_t,ndim=2] C_inv_long = np.zeros((T_num_rows,
                                                          T_num_cols))
    detect_sums = np.zeros((num_T,
                                                            num_E_examples,
                                                            num_detections),dtype=np.float64)
    cdef Py_ssize_t cur_E_idx, cur_T_idx
    for cur_T_idx in range(num_T):
        for cur_E_idx in range(num_E_examples):
            Cs[cur_T_idx,cur_E_idx] = _fast_like2_handmult(Ts[cur_T_idx],
                                        bgs[cur_E_idx],
                                        C_inv_long,
                                        W,
                                        T_num_rows,
                                        T_num_cols,
                                        E_windows[cur_E_idx],
                                        E_num_cols_array[cur_E_idx],
                                        num_detections,
                                        pad_front,
                                        W_bg,
                                        detect_sums[cur_T_idx,cur_E_idx])
    return Cs,detect_sums


def array_fast_like(np.ndarray[dtype_t, ndim=2] T,
                    np.ndarray[dtype_t, ndim=2] bgs,
                    int T_num_rows,
                    int T_num_cols,
                    np.ndarray[dtype_bool_t,ndim=3] E_windows,
                    np.ndarray[int,ndim=1] E_num_cols_array,
                    int num_detections,
                    int pad_front):
    cdef int num_E_examples = E_windows.shape[0]
    Cs = np.zeros(num_E_examples,dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim=1] W_bg = np.zeros(T_num_cols)
    cdef np.ndarray[dtype_t,ndim=2] W = np.zeros((T_num_rows,
                                                  T_num_cols))
    cdef np.ndarray[dtype_t,ndim=2] C_inv_long = np.zeros((T_num_rows,
                                                          T_num_cols))
    cdef np.ndarray[dtype_t,ndim=2] detect_sums = np.zeros((num_E_examples,
                                                            num_detections))
    cdef Py_ssize_t cur_E_idx
    for cur_E_idx in range(num_E_examples):
        Cs[cur_E_idx] = _fast_like2(T,
                                    bgs[cur_E_idx],
                                    C_inv_long,
                                    W,
                                    T_num_rows,
                                    T_num_cols,
                                    E_windows[cur_E_idx],
                                    E_num_cols_array[cur_E_idx],
                                    num_detections,
                                    pad_front,
                                    W_bg,
                                    detect_sums[cur_E_idx])
    return Cs,detect_sums


def fast_like(np.ndarray[dtype_t, ndim=2] T,
              np.ndarray[dtype_t, ndim=1] bg,
              np.ndarray[dtype_t, ndim=2] C_inv_long,
              np.ndarray[dtype_t, ndim=2] W,
              int T_num_rows,
              int T_num_cols,
              np.ndarray[dtype_bool_t,ndim=2] E_window,
              int E_num_cols,
              int num_detections,
              int pad_front,
              np.ndarray[dtype_t,ndim=1] W_bg,
              np.ndarray[dtype_t,ndim=1] detect_sums):
    """
    
    num_detections:
        If this is greater than E_num_cols then we do padding, this is the number of places that we begin
    pad_front:
        how many background vectors to put at front,
        also says how far to start from the beginning,
        we use this to figure out the amount of back padding
        If this is zero then we start all the detections
        from the front of the vector        
    W_bg:
        product vector between the filter W and the
        background, this allows us to compute
        the contribution to the filter sums that comes
        from the padding
    """
    # assumed that this is the height for E_window and the length of bgd
    cdef dtype_t C = compute_W(<dtype_t*> T.data,
                       <dtype_t*> bg.data,
                       <dtype_t*> C_inv_long.data,
                       <dtype_t*> W.data,
                       T_num_rows,
                       T_num_cols)
    # compute the background and W matchup
    dgemv_T(<dtype_t*> W.data,
             <dtype_t*> bg.data,
             <dtype_t*> W_bg.data,
             T_num_rows,
             T_num_cols)
    # computed the pad-matrix
    # f is the frequency/edge type index
    # t is the time, the current place in the
    # num_detections
    # also does the preliminary computation for getting
    # the padding sums
    compute_padding_scores(<dtype_t*> W_bg.data,
                       <dtype_t*> detect_sums.data,
                       num_detections,
                       T_num_cols,
                       pad_front,
                       E_num_cols)
    # we now get the detect sums
    # now we do the sums based on binary features
    filter_data(<dtype_t*> W.data,
                 <dtype_t*> detect_sums.data,
                 <dtype_bool_t*> E_window.data,
                 num_detections,
                 T_num_cols,
                 T_num_rows,
                 pad_front,
                 E_num_cols)
    return C
        

# def matmul(np.ndarray[dtype_t64, ndim=2] A,
#            np.ndarray[dtype_t64, ndim=2] B,
#            np.ndarray[dtype_t64, ndim=2] out):
#     cdef Py_ssize_t i, j
#     cdef np.ndarray[dtype_t64, ndim=1] A_row, B_col
#     for i in range(A.shape[0]):
#         A_row = A[i,:]
#         for j in range(B.shape[1]):
#             B_col = B[:, j]
#             out[i,j] = ddot(
#                 A_row.shape[0],
#                 <dtype_t64*>A_row.data,
#                 A_row.strides[0] // sizeof(dtype_t64),
#                 <dtype_t64*>B_col.data,
#                 B_col.strides[0] // sizeof(dtype_t64))
