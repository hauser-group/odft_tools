import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
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
                 bint dx=False, bint dy=False):
        cdef Py_ssize_t n = X.shape[0]
        cdef Py_ssize_t m = Y.shape[0]
        cdef Py_ssize_t n_dim = X.shape[1]
        cdef Py_ssize_t i, j

        cdef np.ndarray[DTYPE_t, ndim=2] K = np.zeros((n*(1 + int(dx)*n_dim), m*(1 + int(dy)*n_dim)))
        cdef DTYPE_t* Kptr = &K[0, 0]
        
        for i in prange(n, nogil=True):
            for j in range(m):
                if dx and dy:
                    self.calc_element_dx_dy(Kptr, X, Y, i, j, n, m, n_dim)
                elif dx:
                    self.calc_element_dx(Kptr, X, Y, i, j, n, m, n_dim)
                elif dy:
                    self.calc_element_dy(Kptr, X, Y, i, j, n, m, n_dim)
                else:
                    self.calc_element(Kptr, X, Y, i, j, n, m, n_dim)
        return K

    cdef void calc_element(self, DTYPE_t* K, DTYPE_t[:, ::1] X,
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        pass

    cdef void calc_element_dx(self, DTYPE_t* K, DTYPE_t[:, ::1] X,
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        pass

    cdef void calc_element_dy(self, DTYPE_t* K, DTYPE_t[:, ::1] X,
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        pass

    cdef void calc_element_dx_dy(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                                 DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                                 Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        pass

    cdef void calc_element_dx_dy_old(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                                     DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                                     Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        pass

    
cdef class ConvolutionalRBFKernel(Kernel):

    cdef public DTYPE_t length_scale
    cdef public Py_ssize_t p

    def __init__(self, length_scale=1.0, p=2):
        self.length_scale = length_scale
        self.p = 2

    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t g1, g2, q
        cdef DTYPE_t dist, K_ij_g1g2

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q in range(-self.p, self.p + 1):
                    if ((g1 + q) >= 0 and (g1 + q) < n_dim
                            and (g2 + q) >= 0 and (g2 + q) < n_dim):
                        dist += (X[i, g1 + q] - Y[j, g2 + q])**2
                    elif (g1 + q) >= 0 and (g1 + q) < n_dim:
                        dist += X[i, g1 + q]**2
                    elif (g2 + q) >= 0 and (g2 + q) < n_dim:
                        dist += Y[j, g2 + q]**2
                K_ij_g1g2 = exp(-0.5*dist/self.length_scale**2)
                K[i*m + j] += K_ij_g1g2


    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx_dy(self, DTYPE_t* K, DTYPE_t[:, ::1] X,
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t g1, g2, q1, q2
        cdef DTYPE_t dist, K_ij_g1g2, K_ij_g1g2_over_l, K_ij_g1g2_over_l2
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc((2*self.p + 1) * sizeof(DTYPE_t))

        for g1 in range(n_dim):
            for g2 in range(n_dim):
                dist = 0.
                for q1 in range(-self.p, self.p + 1):
                    if ((g1 + q1) >= 0 and (g1 + q1) < n_dim
                            and (g2 + q1) >= 0 and (g2 + q1) < n_dim):
                        scaled_diff[q1 + self.p] = (X[i, g1 + q1] - Y[j, g2 + q1])/self.length_scale
                    elif (g1 + q1) >= 0 and (g1 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = X[i, g1 + q1]/self.length_scale
                    elif (g2 + q1) >= 0 and (g2 + q1) < n_dim:
                        scaled_diff[q1 + self.p] = -Y[j, g2 + q1]/self.length_scale
                    else:
                        scaled_diff[q1 + self.p] = 0.
                    dist += scaled_diff[q1 + self.p]**2
                K_ij_g1g2 = exp(-0.5*dist)
                K_ij_g1g2_over_l = K_ij_g1g2/self.length_scale
                K_ij_g1g2_over_l2 = K_ij_g1g2/self.length_scale**2
                K[i*m*(1+n_dim) + j] += K_ij_g1g2
                for q1 in range(max(-g1, -self.p), min(n_dim-g1, self.p+1)):
                    K[(n + i*n_dim + g1 + q1)*m*(1+n_dim) + j] += -scaled_diff[q1 + self.p]*K_ij_g1g2_over_l
                    for q2 in range(max(-g2, -self.p), min(n_dim-g2, self.p+1)):
                        K[(n + i*n_dim + g1 + q1)*m*(1+n_dim) + m + j*n_dim + g2 + q2] += (
                            - scaled_diff[q1 + self.p] * scaled_diff[q2 + self.p]
                            * K_ij_g1g2_over_l2)
                for q2 in range(-self.p, self.p + 1):
                    if (g2 + q2) >= 0 and (g2 + q2) < n_dim:
                        K[i*m*(1+n_dim) + m + j*n_dim + g2 + q2] += scaled_diff[q2 + self.p]*K_ij_g1g2_over_l
                    if ((g1 + q2) >= 0 and (g1 + q2) < n_dim and
                            (g2 + q2) >= 0 and (g2 + q2) < n_dim):
                        K[(n + i*n_dim + g1 + q2)*m*(1+n_dim) + m + j*n_dim + g2 + q2] += K_ij_g1g2_over_l2
        free(scaled_diff)

    
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
    cdef void calc_element(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t d
        cdef DTYPE_t dist = 0.
        for d in range(n_dim):
            dist += (X[i, d] - Y[j, d])**2
        K[i*m + j] = self.scale * exp(-0.5*dist/self.length_scale**2) + self.constant
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t d
        cdef DTYPE_t tmp, Kij, Kij_over_l, dist = 0.
        for d in range(n_dim):
            tmp = -(X[i, d] - Y[j, d])/self.length_scale
            K[(n + i*n_dim + d)*m + j] = tmp
            dist += tmp*tmp
        Kij = self.scale * exp(-0.5*dist)
        Kij_over_l = Kij/self.length_scale
        K[i*m + j] = Kij + self.constant
        for d in range(n_dim):
            K[(n + i*n_dim + d)*m + j] *= Kij_over_l
    
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dy(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                           DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                           Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t d
        cdef DTYPE_t tmp, Kij, Kij_over_l, dist = 0.
        for d in range(n_dim):
            tmp = (X[i, d] - Y[j, d])/self.length_scale
            K[i*m*(1+n_dim) + m + j*n_dim + d] = tmp
            dist += tmp*tmp
        Kij = self.scale * exp(-0.5*dist)
        Kij_over_l = Kij/self.length_scale
        K[i*m*(1+n_dim) + j] = Kij + self.constant
        for d in range(n_dim):
            K[i*m*(1+n_dim) + m + j*n_dim + d] *= Kij_over_l
            
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx_dy(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                                 DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                                 Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        cdef Py_ssize_t d1, d2
        cdef DTYPE_t Kij, Kij_over_l2, Kij_over_l, dist = 0.
        
        cdef DTYPE_t *scaled_diff = <DTYPE_t *> malloc(n_dim * sizeof(DTYPE_t))
        
        for d1 in range(n_dim):
            scaled_diff[d1] = (X[i, d1] - Y[j, d1])/self.length_scale
            dist += scaled_diff[d1]*scaled_diff[d1]
        Kij = self.scale*exp(-0.5*dist)
        Kij_over_l = Kij/self.length_scale
        Kij_over_l2 = Kij/self.length_scale**2
        K[i*m*(1+n_dim) + j] = Kij + self.constant
        
        for d1 in range(n_dim):
            K[i*m*(1+n_dim) + m + j*n_dim + d1] = scaled_diff[d1]*Kij_over_l
            K[(n + i*n_dim + d1)*m*(1+n_dim) + j] = -scaled_diff[d1]*Kij_over_l
            for d2 in range(n_dim):
                K[(n + i*n_dim + d1)*m*(1+n_dim) + m + j*n_dim + d2] = (
                    -scaled_diff[d1]*scaled_diff[d2])*Kij_over_l2
            K[(n + i*n_dim + d1)*m*(1+n_dim) + m + j*n_dim + d1] += Kij_over_l2
        
        free(scaled_diff)
        
    @cython.cdivision(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void calc_element_dx_dy_old(self, DTYPE_t* K, DTYPE_t[:, ::1] X, 
                                     DTYPE_t[:, ::1] Y, Py_ssize_t i, Py_ssize_t j,
                                     Py_ssize_t n, Py_ssize_t m, Py_ssize_t n_dim) nogil:
        """While this version is slightly slower it comes by without using malloc
           and free."""
        cdef Py_ssize_t d1, d2
        cdef DTYPE_t Kij, Kij_over_l2, scaled_diff, dist = 0.
        for d1 in range(n_dim):
            dist += (X[i, d1] - Y[j, d1])**2
        Kij = self.scale*exp(-0.5*dist/self.length_scale**2)
        Kij_over_l2 = Kij/self.length_scale**2
        K[i*m*(1+n_dim) + j] = Kij + self.constant
        for d1 in range(n_dim):
            # Slightly different than the typical definition since
            # the difference is scaled by l^2
            scaled_diff = (X[i, d1] - Y[j, d1])/self.length_scale**2
            K[i*m*(1+n_dim) + m + j*n_dim + d1] = scaled_diff*Kij
            K[(n + i*n_dim + d1)*m*(1+n_dim) + j] = -scaled_diff*Kij
            for d2 in range(n_dim):
                K[(n + i*n_dim + d1)*m*(1+n_dim) + m + j*n_dim + d2] = (
                    # Note that both factors 1/l are already taken into 
                    # account by scaled_diff 
                    -scaled_diff*(X[i, d2] - Y[j, d2]))*Kij_over_l2
            K[(n + i*n_dim + d1)*m*(1+n_dim) + m + j*n_dim + d1] += Kij_over_l2
