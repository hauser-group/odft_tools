import numpy as np
import unittest
from odft_tools.optimization import PCA, SineBasis

class ProjectionTest():
    class ProjectionTest(unittest.TestCase):
        
        def test_idempotence(self):
            x_test = np.random.randn(51)
            P = self.projection(x_test)
            
            np.testing.assert_allclose(P, P.dot(P), atol=1e-10)
            
            
class SineBasisTest(ProjectionTest.ProjectionTest):
    projection = SineBasis(G=51, h=1, l=11)
        
class PCATest(ProjectionTest.ProjectionTest):
    projection = PCA(np.random.randn(100, 51), m=30, l=5)
    
if __name__ == '__main__':
    unittest.main() 