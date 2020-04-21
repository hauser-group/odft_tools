import numpy as np
import unittest
from odft_tools.utils import first_derivative_matrix, second_derivative_matrix


class DerivativeTest(unittest.TestCase):
    
    def fun(self, x):
        return np.sin(2*np.pi*x)
    
    def dfun(self, x):
        return 2*np.pi*np.cos(2*np.pi*x)
    
    def d2fun(self, x):
        return -4*np.pi**2*np.sin(2*np.pi*x)
    
    def eval_first_deriv_errors(self, method):
        Gs = [51, 101, 201, 401, 801]
        abs_errors = np.zeros(len(Gs))
        for i, G in enumerate(Gs):
            x_vec = np.linspace(0, 1, G)
            h = x_vec[1] - x_vec[0]
            mat = first_derivative_matrix(G, h, method=method)

            y = self.fun(x_vec)
            dy = self.dfun(x_vec)
            dy_num = mat.dot(y)
            abs_errors[i] = np.max(np.abs(dy - dy_num))
        return abs_errors
    
    def test_first_deriv_matrix_three_point(self):        
        # reducing the step size by a factor of 2
        # should reduce the error by a factor of 4
        # since the 3 point method is O(h^2).
        # Only checking for reduction > 3.9:
        abs_errors = self.eval_first_deriv_errors('three_point')
        self.assertTrue(np.all(abs_errors[0:-1] / abs_errors[1:] > 3.9))
        
    def test_first_deriv_matrix_five_point(self):
        # reducing the step size by a factor of 2
        # should reduce the error by a factor of 16
        # since the 5 point method is O(h^4).
        # Only checking for reduction > 15:
        abs_errors = self.eval_first_deriv_errors('five_point')
        self.assertTrue(np.all(abs_errors[0:-1] / abs_errors[1:] > 15))
        
    def eval_second_deriv_errors(self, method):
        Gs = [51, 101, 201, 401, 801]
        abs_errors = np.zeros(len(Gs))
        for i, G in enumerate(Gs):
            x_vec = np.linspace(0, 1, G)
            h = x_vec[1] - x_vec[0]
            mat = second_derivative_matrix(G, h, method=method)

            y = self.fun(x_vec)
            d2y = self.d2fun(x_vec)
            d2y_num = mat.dot(y)
            abs_errors[i] = np.max(np.abs(d2y - d2y_num))
        return abs_errors
    
    def test_second_deriv_matrix_three_point(self):        
        # reducing the step size by a factor of 2
        # should reduce the error by a factor of 4
        # since the 3 point method is O(h^2).
        # Only checking for reduction > 3.9:
        abs_errors = self.eval_second_deriv_errors('three_point')
        print(abs_errors[0:-1] / abs_errors[1:])
        self.assertTrue(np.all(abs_errors[0:-1] / abs_errors[1:] > 3.9))
        
    def test_second_deriv_matrix_five_point(self):
        # reducing the step size by a factor of 2
        # should reduce the error by a factor of 16
        # since the 5 point method is O(h^4).
        # Only checking for reduction > 15:
        abs_errors = self.eval_second_deriv_errors('five_point')
        print(abs_errors[0:-1] / abs_errors[1:])
        self.assertTrue(np.all(abs_errors[0:-1] / abs_errors[1:] > 15))
       
        
if __name__ == '__main__':
    unittest.main()     