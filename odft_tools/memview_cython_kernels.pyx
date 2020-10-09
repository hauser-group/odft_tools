import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, sqrt
from libc.stdlib cimport malloc, free
from cython.parallel import prange

ctypedef np.float_t DTYPE_t

cdef class Kernel():
    """Base class for all kernels"""
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, 
                 DTYPE_t[:, ::1] X, 
                 DTYPE_t[:, ::1] Y, 
                 bint dx=False, bint dy=False, DTYPE_t h=1.0):
        cdef Py_ssize_t n = X.shape[0]
        cdef Py_ssize_t m = Y.shape[0]
        cdef Py_ssize_t n_dim = X.shape[1]
        cdef Py_ssize_t i, j
        
        cdef np.ndarray[DTYPE_t, ndim=2] K = np.zeros((n*(1 + int(dx)*n_dim), m*(1 + int(dy)*n_dim)))
        cdef DTYPE_t[:, ::1] Kview = K
        
        if dx and dy:
            for i in prange(n, nogil=True):
                for j in range(m):                
                    self.calc_element_dx_dy(
                        &Kview[i, j],
                        Kview[n + i*n_dim:n + (i+1)*n_dim, j],
                        Kview[i, m + j*n_dim:m + (j+1)*n_dim],
                        Kview[n + i*n_dim:n + (i+1)*n_dim, 
                              m + j*n_dim:m + (j+1)*n_dim],
                        X[i], Y[j], n_dim, h)
        elif dx:
            for i in prange(n, nogil=True):
                for j in range(m): 
                    self.calc_element_dx(
                        &Kview[i, j],
                        Kview[n + i*n_dim:n + (i+1)*n_dim, j],
                        X[i], Y[j], n_dim, h)
        elif dy:
            for i in prange(n, nogil=True):
                for j in range(m): 
                    self.calc_element_dy(
                        &Kview[i, j],
                        Kview[i, m + j*n_dim:m + (j+1)*n_dim],
                        X[i], Y[j], n_dim, h)
        else:
            for i in prange(n, nogil=True):
                for j in range(m): 
                    self.calc_element(&Kview[i, j], X[i], Y[j], n_dim, h)                        
        return K
    
    cdef void calc_element(
            self, DTYPE_t[] K, DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        pass
    
    cdef void calc_element_dx(
            self, DTYPE_t[] K, DTYPE_t[:] J, DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        pass
    
    cdef void calc_element_dy(
            self, DTYPE_t[] K, DTYPE_t[:] J_prime, DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        pass
    
    cdef void calc_element_dx_dy(self, DTYPE_t[] K, DTYPE_t[:] J, DTYPE_t[:] J_prime, DTYPE_t[:, :] H,
                                 DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        pass

    
cdef class RBFKernel(Kernel):
    
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
    cdef void calc_element(self, DTYPE_t[] K, DTYPE_t[::1] X, 
                              DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t d
        cdef DTYPE_t dist = 0.
        for d in range(n_dim):
            dist += (X[d] - Y[d])**2
        K[0] = self.scale * exp(-0.5*dist/self.length_scale**2) + self.constant
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx(
            self, DTYPE_t[] K, DTYPE_t[:] J, DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t d
        cdef DTYPE_t tmp, Kij, Kij_over_l, dist = 0.
        for d in range(n_dim):
            tmp = -(X[d] - Y[d])/self.length_scale
            J[d] = tmp
            dist += tmp*tmp
        Kij = self.scale * exp(-0.5*dist)
        K[0] = Kij + self.constant
        Kij_over_l = Kij/(self.length_scale*h) 
        for d in range(n_dim):
            J[d] *= Kij_over_l
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dy(
            self, DTYPE_t[] K, DTYPE_t[:] J_prime, DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t d
        cdef DTYPE_t tmp, Kij, Kij_over_l, dist = 0.
        for d in range(n_dim):
            tmp = (X[d] - Y[d])/self.length_scale
            J_prime[d] = tmp
            dist += tmp*tmp
        Kij = self.scale * exp(-0.5*dist)
        K[0] = Kij + self.constant
        Kij_over_l = Kij/(self.length_scale*h)
        for d in range(n_dim):
            J_prime[d] *= Kij_over_l
            
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx_dy(self, DTYPE_t[] K, DTYPE_t[:] J, DTYPE_t[:] J_prime, DTYPE_t[:, :] H,
                                 DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t d1, d2
        cdef DTYPE_t Kij, Kij_over_l2, Kij_over_l, dist = 0.
        
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc(n_dim * sizeof(DTYPE_t))
        
        for d1 in range(n_dim):
            scaled_diff[d1] = (X[d1] - Y[d1])/self.length_scale
            dist += scaled_diff[d1]*scaled_diff[d1]
        Kij = self.scale*exp(-0.5*dist)
        K[0] = Kij + self.constant
        Kij_over_l = Kij/(self.length_scale*h)
        Kij_over_l2 = Kij/(self.length_scale*h)**2
        
        for d1 in range(n_dim):
            J_prime[d1] = scaled_diff[d1]*Kij_over_l
            J[d1] = -scaled_diff[d1]*Kij_over_l
            for d2 in range(n_dim):
                H[d1, d2] = -scaled_diff[d1]*scaled_diff[d2]*Kij_over_l2
            H[d1, d1] += Kij_over_l2
        
        free(scaled_diff)
    
cdef class ConvolutionalRBFKernel(Kernel):

    cdef public DTYPE_t length_scale
    cdef public Py_ssize_t p

    def __init__(self, length_scale=1.0, p=2):
        self.length_scale = length_scale
        self.p = 2

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element(self, DTYPE_t[] K, DTYPE_t[::1] X, 
                           DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t g1, g2, q
        cdef DTYPE_t dist

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q in range(-self.p, self.p + 1):
                    if ((g1 + q) >= 0 and (g1 + q) < n_dim
                            and (g2 + q) >= 0 and (g2 + q) < n_dim):
                        dist += (X[g1 + q] - Y[g2 + q])**2
                    elif (g1 + q) >= 0 and (g1 + q) < n_dim:
                        dist += X[g1 + q]**2
                    elif (g2 + q) >= 0 and (g2 + q) < n_dim:
                        dist += Y[g2 + q]**2
                K[0] += exp(-0.5*dist/self.length_scale**2)
        
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx(self, DTYPE_t[] K, DTYPE_t[:] J,
                              DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t g1, g2, q1
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l, K_ij_g1g2_over_l2
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[g1 + q1] - Y[g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/(self.length_scale*h)
                K[0] += K_ij_g1g2
                for q1 in range(max(-g1, -self.p), min(n_dim-g1, self.p+1)):
                    J[g1 + q1] += -scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
        free(scaled_diff)

    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dy(self, DTYPE_t[] K, DTYPE_t[:] J_prime,
                              DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t g1, g2, q1
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[g1 + q1] - Y[g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/(self.length_scale*h)
                K[0] += K_ij_g1g2
                for q1 in range(max(-g2, -self.p), min(n_dim-g2, self.p+1)):
                    J_prime[g2 + q1] += scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
        free(scaled_diff)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx_dy(self, DTYPE_t[] K, DTYPE_t[:] J, DTYPE_t[:] J_prime, DTYPE_t[:, :] H,
                                 DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim, DTYPE_t h) nogil:
        cdef Py_ssize_t g1, g2, q1, q2
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l, K_ij_g1g2_over_l2
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[g1 + q1] - Y[g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/(self.length_scale*h)
                K_ij_g1g2_over_l2 = K_ij_g1g2/(self.length_scale*h)**2
                K[0] += K_ij_g1g2
                for q1 in range(max(-g1, -self.p), min(n_dim-g1, self.p+1)):
                    J[g1 + q1] += -scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
                    for q2 in range(max(-g2, -self.p), min(n_dim-g2, self.p+1)):
                        H[g1 + q1, g2 + q2] += (
                            - scaled_diff[q1 + self.p] * scaled_diff[q2 + self.p]
                            * K_ij_g1g2_over_l2)
                for q2 in range(-self.p, self.p + 1):
                    if (g2 + q2) >= 0 and (g2 + q2) < n_dim:
                        J_prime[g2 + q2] += scaled_diff[q2 + self.p]*K_ij_g1g2_over_l
                    if ((g1 + q2) >= 0 and (g1 + q2) < n_dim and
                            (g2 + q2) >= 0 and (g2 + q2) < n_dim):
                        H[g1 + q2, g2 + q2] += K_ij_g1g2_over_l2
        free(scaled_diff)
        
        
cdef class NormalizedConvolutionalRBFKernel():

    cdef public DTYPE_t length_scale
    cdef public Py_ssize_t p

    def __init__(self, length_scale=1.0, p=2):
        self.length_scale = length_scale
        self.p = 2
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, 
                 DTYPE_t[:, ::1] X, 
                 DTYPE_t[:, ::1] Y, 
                 bint dx=False, bint dy=False, DTYPE_t h=1.0):
        cdef Py_ssize_t n = X.shape[0]
        cdef Py_ssize_t m = Y.shape[0]
        cdef Py_ssize_t n_dim = X.shape[1]
        cdef Py_ssize_t i, j, d1, d2
        cdef DTYPE_t norm_xy
        
        cdef np.ndarray[DTYPE_t, ndim=2] K = np.zeros((n*(1 + int(dx)*n_dim), m*(1 + int(dy)*n_dim)))
        cdef DTYPE_t[:, ::1] Kview = K
        
        cdef DTYPE_t[::1] K_xx = np.zeros(n*(1 + int(dx)*n_dim))
        cdef DTYPE_t[::1] K_yy = np.zeros(m*(1 + int(dy)*n_dim))
        
        if dx and dy:
            for i in prange(n, nogil=True):
                self.calc_element_dx(
                    &K_xx[i], 
                    K_xx[n + i*n_dim:n + (i+1)*n_dim], 
                    X[i], X[i], n_dim)
                
            for j in prange(m, nogil=True):
                self.calc_element_dx(
                    &K_yy[j], 
                    K_yy[m + j*n_dim:m + (j+1)*n_dim], 
                    Y[j], Y[j], n_dim)
            
            for i in range(n):
                for j in range(m):       
                    norm_xy = sqrt(K_yy[j])*sqrt(K_xx[i])
                    self.calc_element_dx_dy(
                        &Kview[i, j],
                        Kview[n + i*n_dim:n + (i+1)*n_dim, j],
                        Kview[i, m + j*n_dim:m + (j+1)*n_dim],
                        Kview[n + i*n_dim:n + (i+1)*n_dim, 
                              m + j*n_dim:m + (j+1)*n_dim],
                        X[i], Y[j], n_dim)
                    for d1 in range(n_dim):
                        for d2 in range(n_dim):
                            Kview[n + i*n_dim + d1, m + j*n_dim + d2] += (
                                Kview[i, j]*K_xx[n + i*n_dim + d1]*K_yy[m + j*n_dim + d2]/(K_xx[i]*K_yy[j]))
                            Kview[n + i*n_dim + d1, m + j*n_dim + d2] -= (
                                Kview[n + i*n_dim + d1, j]*K_yy[m + j*n_dim + d2])/K_yy[j]
                            Kview[n + i*n_dim + d1, m + j*n_dim + d2] -= (
                                K_xx[n + i*n_dim + d1]*Kview[i, m + j*n_dim + d2])/K_xx[i]
                            Kview[n + i*n_dim + d1, m + j*n_dim + d2] /= norm_xy*h*h
                    for d1 in range(n_dim):
                        Kview[n + i*n_dim + d1, j] -= K_xx[n + i*n_dim + d1]*Kview[i, j]/(K_xx[i])
                        Kview[n + i*n_dim + d1, j] /= norm_xy*h
                        Kview[i, m + j*n_dim + d1] -= K_yy[m + j*n_dim + d1]*Kview[i, j]/(K_yy[j])
                        Kview[i, m + j*n_dim + d1] /= norm_xy*h                      
                    Kview[i, j] /= norm_xy
        elif dx:
            for i in prange(n, nogil=True):
                self.calc_element_dx(
                    &K_xx[i], 
                    K_xx[n + i*n_dim:n + (i+1)*n_dim], 
                    X[i], X[i], n_dim)
                
            for j in prange(m, nogil=True):
                self.calc_element(
                    &K_yy[j], Y[j], Y[j], n_dim)

            for i in prange(n, nogil=True):
                for j in range(m): 
                    norm_xy = sqrt(K_yy[j])*sqrt(K_xx[i])
                    self.calc_element_dx(
                        &Kview[i, j],
                        Kview[n + i*n_dim:n + (i+1)*n_dim, j], 
                        X[i], Y[j], n_dim)
                    for d1 in range(n_dim):
                        Kview[n + i*n_dim + d1, j] -= K_xx[n + i*n_dim + d1]*Kview[i, j]/(K_xx[i])
                        Kview[n + i*n_dim + d1, j] /= norm_xy*h                     
                    Kview[i, j] /= norm_xy
        elif dy:
            for i in prange(n, nogil=True):
                self.calc_element(
                    &K_xx[i], X[i], X[i], n_dim)
                
            for j in prange(m, nogil=True):
                self.calc_element_dx(
                    &K_yy[j], 
                    K_yy[m + j*n_dim:m + (j+1)*n_dim], 
                    Y[j], Y[j], n_dim)
    
            for i in prange(n, nogil=True):
                for j in range(m):
                    norm_xy = sqrt(K_yy[j])*sqrt(K_xx[i])
                    self.calc_element_dy(
                        &Kview[i, j],
                        Kview[i, m + j*n_dim:m + (j+1)*n_dim], 
                        X[i], Y[j], n_dim)
                    for d1 in range(n_dim):
                        Kview[i, m + j*n_dim + d1] -= K_yy[m + j*n_dim + d1]*Kview[i, j]/(K_yy[j])
                        Kview[i, m + j*n_dim + d1] /= norm_xy*h                        
                    Kview[i, j] /= norm_xy
        else:
            for i in prange(n, nogil=True):
                self.calc_element(
                    &K_xx[i], X[i], X[i], n_dim)
                
            for j in prange(m, nogil=True):
                self.calc_element(
                    &K_yy[j], Y[j], Y[j], n_dim)

            for i in prange(n, nogil=True):
                for j in range(m):
                    norm_xy = sqrt(K_yy[j])*sqrt(K_xx[i])
                    self.calc_element(&Kview[i, j], X[i], Y[j], n_dim)         
                    Kview[i, j] /= norm_xy
        return K

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element(self, DTYPE_t[] K, DTYPE_t[::1] X, 
                           DTYPE_t[::1] Y, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t g1, g2, q
        cdef DTYPE_t dist

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q in range(-self.p, self.p + 1):
                    if ((g1 + q) >= 0 and (g1 + q) < n_dim
                            and (g2 + q) >= 0 and (g2 + q) < n_dim):
                        dist += (X[g1 + q] - Y[g2 + q])**2
                    elif (g1 + q) >= 0 and (g1 + q) < n_dim:
                        dist += X[g1 + q]**2
                    elif (g2 + q) >= 0 and (g2 + q) < n_dim:
                        dist += Y[g2 + q]**2
                K[0] += exp(-0.5*dist/self.length_scale**2)
        
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx(self, DTYPE_t[] K, DTYPE_t[:] J,
                              DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t g1, g2, q1
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l, K_ij_g1g2_over_l2
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[g1 + q1] - Y[g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/self.length_scale
                K[0] += K_ij_g1g2
                for q1 in range(max(-g1, -self.p), min(n_dim-g1, self.p+1)):
                    J[g1 + q1] += -scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
        free(scaled_diff)

    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dy(self, DTYPE_t[] K, DTYPE_t[:] J_prime,
                              DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t g1, g2, q1
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[g1 + q1] - Y[g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/self.length_scale
                K[0] += K_ij_g1g2
                for q1 in range(max(-g2, -self.p), min(n_dim-g2, self.p+1)):
                    J_prime[g2 + q1] += scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
        free(scaled_diff)


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx_dy(self, DTYPE_t[] K, DTYPE_t[:] J, DTYPE_t[:] J_prime, DTYPE_t[:, :] H,
                                 DTYPE_t[::1] X, DTYPE_t[::1] Y, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t g1, g2, q1, q2
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l, K_ij_g1g2_over_l2
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[g1 + q1] - Y[g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/self.length_scale
                K_ij_g1g2_over_l2 = K_ij_g1g2/self.length_scale**2
                K[0] += K_ij_g1g2
                for q1 in range(max(-g1, -self.p), min(n_dim-g1, self.p+1)):
                    J[g1 + q1] += -scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
                    for q2 in range(max(-g2, -self.p), min(n_dim-g2, self.p+1)):
                        H[g1 + q1, g2 + q2] += (
                            - scaled_diff[q1 + self.p] * scaled_diff[q2 + self.p]
                            * K_ij_g1g2_over_l2)
                for q2 in range(-self.p, self.p + 1):
                    if (g2 + q2) >= 0 and (g2 + q2) < n_dim:
                        J_prime[g2 + q2] += scaled_diff[q2 + self.p]*K_ij_g1g2_over_l
                    if ((g1 + q2) >= 0 and (g1 + q2) < n_dim and
                            (g2 + q2) >= 0 and (g2 + q2) < n_dim):
                        H[g1 + q2, g2 + q2] += K_ij_g1g2_over_l2
        free(scaled_diff)