import numpy as np
cimport numpy as np
cimport cython
import multiprocessing
from cython.parallel import prange, threadid
from libc.math cimport exp

ctypedef np.float_t DTYPE_t

cdef class RBFKernel:
    
    cdef public DTYPE_t length_scale
    cdef public DTYPE_t scale
    cdef public DTYPE_t constant    
    
    def __init__(self, length_scale=1.0, scale=1.0, constant=0.0):
        self.length_scale = length_scale
        self.scale = scale
        self.constant = constant

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False) 
    def __call__(self, 
                 DTYPE_t[:,:] X,
                 DTYPE_t[:,:] Y,
                 bint dx=False, bint dy=False, 
                 DTYPE_t h=1.0):
        cdef Py_ssize_t num_threads = multiprocessing.cpu_count()
        cdef Py_ssize_t n = X.shape[0]
        cdef Py_ssize_t m = Y.shape[0]
        cdef Py_ssize_t n_dim = X.shape[1]
        cdef Py_ssize_t i, j, d, d2, tid
        cdef DTYPE_t dist, Kij_over_l, Kij_over_l2
        cdef DTYPE_t[:,:] diff = np.zeros((num_threads, n_dim))
        cdef np.ndarray[DTYPE_t, ndim=2] K = np.zeros((n*(1 + int(dx)*n_dim),
                                                       m*(1 + int(dy)*n_dim)))
        for i in prange(n, nogil=True, num_threads=num_threads):
            tid = threadid()
            for j in range(m):
                dist = 0.
                for d in range(n_dim):
                    diff[tid, d] = (X[i, d] - Y[j, d])/self.length_scale
                    dist = dist + diff[tid, d]*diff[tid, d]
                K[i, j] = self.scale * exp(-0.5*dist)
                Kij_over_l = K[i, j]/(self.length_scale*h)
                Kij_over_l2 = Kij_over_l/(self.length_scale*h)
                if dy and not dx:
                    for d in range(n_dim):
                        K[i, m + j*n_dim + d] = diff[tid, d]*Kij_over_l
                if dx and not dy:
                    for d in range(n_dim):
                        K[n + i*n_dim + d, j] = -diff[tid, d]*Kij_over_l
                if dx and dy:
                    for d in range(n_dim):
                        K[i, m + j*n_dim + d] = diff[tid, d]*Kij_over_l
                        K[n + i*n_dim + d, j] = -K[i, m + j*n_dim + d]
                        for d2 in range(n_dim):
                            K[n + i*n_dim + d, m + j*n_dim + d2] = (
                                - diff[tid, d]*diff[tid, d2])*Kij_over_l2
                        # Add the diagonal element d==d2
                        K[n + i*n_dim + d, m + j*n_dim + d] = (
                            K[n + i*n_dim + d, m + j*n_dim + d] + Kij_over_l2)
                K[i, j] = K[i, j] + self.constant
        return K