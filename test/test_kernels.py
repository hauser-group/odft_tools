import numpy as np
import unittest
from odft_tools.kernels import RBFKernel
from odft_tools.cython_kernels import RBFKernel as RBFKernel_cy
from odft_tools.memview_cython_kernels import RBFKernel as RBFKernel_memview_cy
from odft_tools.pointer_cython_kernels import RBFKernel as RBFKernel_pointer_cy

class KernelTest():
    class KernelTest(unittest.TestCase):
        # Default value for the dimension, can be overwritten in subclasses
        n_dim = 7
        
        @staticmethod
        def numerical_derivatives_3point(fun, x0, y0, dx=1e-4, dy=1e-4):
            n_dim = x0.shape[0]
            J = np.zeros(n_dim)
            J_prime = np.zeros(n_dim)
            H = np.zeros((n_dim, n_dim))

            for i in range(n_dim):
                dx_vec = np.zeros(n_dim)
                dx_vec[i] += dx
                J[i] = (fun(x0 + dx_vec, y0) - fun(x0 - dx_vec, y0))/(2*dx)
                
                for j in range(n_dim):
                    dy_vec = np.zeros(n_dim)
                    dy_vec[j] += dy
                    if i == 0:
                        J_prime[j] = (fun(x0, y0 + dy_vec)
                                      - fun(x0, y0 - dy_vec))/(2*dy)
                    H[i, j] = (fun(x0 + dx_vec, y0 + dy_vec)
                               - fun(x0 + dx_vec, y0 - dy_vec)
                               - fun(x0 - dx_vec, y0 + dy_vec)
                               + fun(x0 - dx_vec, y0 - dy_vec)
                              )/(4*dx*dy)
                
            return J, J_prime, H
        
        @staticmethod
        def numerical_derivatives_5point(fun, x0, y0, dx=1e-4, dy=1e-4):
            n_dim = x0.shape[0]
            J = np.zeros(n_dim)
            J_prime = np.zeros(n_dim)
            H = np.zeros((n_dim, n_dim))

            for i in range(n_dim):
                dx_vec = np.zeros(n_dim)
                dx_vec[i] += dx
                J[i] = (fun(x0 - 2*dx_vec, y0) 
                        - 8*fun(x0 - dx_vec, y0)
                        + 8*fun(x0 + dx_vec, y0)
                        - fun(x0 + 2*dx_vec, y0))/(12*dx)
                
                for j in range(n_dim):
                    dy_vec = np.zeros(n_dim)
                    dy_vec[j] += dy
                    if i == 0:
                        J_prime[j] = (fun(x0, y0 + dy_vec)
                                      - fun(x0, y0 - dy_vec))/(2*dy)
                    H[i, j] = (fun(x0 - 2*dx_vec, y0 - 2*dy_vec)
                               - 8*fun(x0 - dx_vec, y0 - 2*dy_vec)
                               + 8*fun(x0 + dx_vec, y0 - 2*dy_vec)
                               - fun(x0 + 2*dx_vec, y0 - 2*dy_vec)
                               - 8*fun(x0 - 2*dx_vec, y0 - dy_vec)
                               + 8*8*fun(x0 - dx_vec, y0 - dy_vec)
                               - 8*8*fun(x0 + dx_vec, y0 - dy_vec)
                               + 8*fun(x0 + 2*dx_vec, y0 - dy_vec)
                               + 8*fun(x0 - 2*dx_vec, y0 + dy_vec)
                               - 8*8*fun(x0 - dx_vec, y0 + dy_vec)
                               + 8*8*fun(x0 + dx_vec, y0 + dy_vec)
                               - 8*fun(x0 + 2*dx_vec, y0 + dy_vec)
                               - fun(x0 - 2*dx_vec, y0 + 2*dy_vec)
                               + 8*fun(x0 - dx_vec, y0 + 2*dy_vec)
                               - 8*fun(x0 + dx_vec, y0 + 2*dy_vec)
                               + fun(x0 + 2*dx_vec, y0 + 2*dy_vec)
                              )/(12*12*dx*dy)
                
            return J, J_prime, H
        
        @staticmethod
        def numerical_derivatives_old(fun, x0, y0, dx=1e-4, dy=1e-4):
            """ Follows the idea from 
            https://en.wikipedia.org/wiki/Finite_difference#Finite_difference_in_several_variables
            to calculate both pure and the mixed derivative in a single function"""
            n_dim = x0.shape[0]
            J = np.zeros(n_dim)
            J_prime = np.zeros(n_dim)
            H = np.zeros((n_dim, n_dim))

            f0 = fun(x0, y0)
            f_x_plus = np.zeros(n_dim)
            f_x_minus = np.zeros(n_dim)
            f_y_plus = np.zeros(n_dim)
            f_y_minus = np.zeros(n_dim)
            for i in range(n_dim):
                dx_vec = np.zeros(n_dim)
                dx_vec[i] += dx
                f_x_plus[i] = fun(x0 + dx_vec, y0)
                f_x_minus[i] = fun(x0 - dx_vec, y0)
                
                for j in range(n_dim):
                    dy_vec = np.zeros(n_dim)
                    dy_vec[j] += dy
                    if i == 0:
                        f_y_plus[j] = fun(x0, y0 + dy_vec)
                        f_y_minus[j] = fun(x0, y0 - dy_vec)
                    f_x_plus_y_plus = fun(x0 + dx_vec, y0 + dy_vec)
                    f_x_minus_y_minus = fun(x0 - dx_vec, y0 - dy_vec)
                    H[i, j] = (f_x_plus_y_plus - f_x_plus[i]
                               - f_y_plus[j] + 2*f0 - f_x_minus[i]
                               - f_y_minus[j] + f_x_minus_y_minus
                              )/(2*dx*dy)

            J = (f_x_plus - f_x_minus)/(2*dx)
            J_prime = (f_y_plus - f_y_minus)/(2*dy)
                
            return J, J_prime, H
        
        @staticmethod
        def numerical_vector_derivative(fun, x0, dx=1e-4):
            """Allows differentiation of functions that return vectors or matrices"""
            n_dim = x0.shape[0]
            f0 = fun(x0)
            deriv = np.zeros((*f0.shape, n_dim))            

            for i in range(n_dim):
                dx_vec = np.zeros(n_dim)
                dx_vec[i] += dx
                deriv[..., i] = (fun(x0 - 2*dx_vec) 
                                 - 8*fun(x0 - dx_vec)
                                 + 8*fun(x0 + dx_vec)
                                 - fun(x0 + 2*dx_vec))/(12*dx)
                
            return deriv
        
        
        def test_numerical_derivative(self):
            n = 4
            m = 3
            np.random.seed(0)
            X = np.random.randn(n, self.n_dim)
            Y = np.random.randn(m, self.n_dim)

            K_ref = self.kernel(X, Y, dx=True, dy=True)
            J_ref = K_ref[n:, :m]
            J_num = np.zeros((n*self.n_dim, m))
            J_prime_ref = K_ref[:n, m:]
            J_prime_num = np.zeros((n, m*self.n_dim))
            H_ref = K_ref[n:, m:]
            H_num = np.zeros((n*self.n_dim, m*self.n_dim))

            J_fun = lambda Xi: self.kernel(Xi[np.newaxis, :], Y)[0, :m]
            J_prime_fun = lambda Yj: self.kernel(X, Yj[np.newaxis, :])[:n, 0]
            H_fun = lambda Xi: self.kernel(Xi[np.newaxis, :], Y, dy=True)[0, m:]
            #H_fun2 =  lambda Yj: self.kernel(X, Yj[np.newaxis, :], dx=True)[n:, 0]

            for i in range(n):
                di = slice(i*self.n_dim, (i+1)*self.n_dim, 1) 
                J_num[di, :m] = self.numerical_vector_derivative(J_fun, X[i]).T
                H_num[di, :] = self.numerical_vector_derivative(H_fun, X[i]).T
                
            for j in range(m):
                dj = slice(j*self.n_dim, (j+1)*self.n_dim, 1)
                J_prime_num[:n, dj] = self.numerical_vector_derivative(J_prime_fun, Y[j])
                #H_num[:, dj] = self.numerical_vector_derivative(H_fun2, Y[j])                   
                    
            np.testing.assert_allclose(J_ref, J_num, atol=1e-10)
            np.testing.assert_allclose(J_prime_ref, J_prime_num, atol=1e-10)
            np.testing.assert_allclose(H_ref, H_num, atol=1e-10)
            
        def test_call_method(self):
            """Tests if the different kwargs to the __call__ method, i.e. dx and dy
            give consistent results"""
            n = 4
            m = 3
            np.random.seed(0)
            X = np.random.randn(n, self.n_dim)
            Y = np.random.randn(m, self.n_dim)
            
            K = self.kernel(X, Y, dx=True, dy=True)
            K_ref = K[:n, :m]
            J_ref = K[n:, :m]
            J_prime_ref = K[:n, m:]
            
            K_test = self.kernel(X, Y, dx=True, dy=False)
            np.testing.assert_allclose(K[:n, :m], K_ref)
            np.testing.assert_allclose(K[n:, :m], J_ref)
            
            K_test = self.kernel(X, Y, dx=False, dy=True)
            np.testing.assert_allclose(K[:n, :m], K_ref)
            np.testing.assert_allclose(K[:n, m:], J_prime_ref)
            
class RBFKernelTest(KernelTest.KernelTest):
    kernel = RBFKernel(length_scale=2.2, scale=1.2, constant=0.6)
    n_dim = 15
    
class CythonRBFKernelTest(KernelTest.KernelTest):
    kernel = RBFKernel_cy(length_scale=2.2, scale=1.2, constant=0.6)
    n_dim = 15
    
class MemviewCythonRBFKernelTest(KernelTest.KernelTest):
    kernel = RBFKernel_memview_cy(length_scale=2.2, scale=1.2, constant=0.6)
    n_dim = 15

class PointerCythonRBFKernelTest(KernelTest.KernelTest):
    kernel = RBFKernel_pointer_cy(length_scale=2.2, scale=1.2, constant=0.6)
    n_dim = 15
    
if __name__ == '__main__':
    unittest.main() 
