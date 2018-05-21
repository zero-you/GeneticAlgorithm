import unittest
import numpy as np

from ga import SimpleGeneticAlgorithm

class TestGA(unittest.TestCase):

    def test_sumSquare(self):
        def sumSquare(x):
            return np.sum(x**2, axis=1)

        x_bounds = np.array([
            [-2, 2],
            [-2, 2],
            [-1, 1]])

        ga_opt = SimpleGeneticAlgorithm(sumSquare, x_bounds, verbose=1)
        ga_opt.minimize()

        self.assertTrue(ga_opt.fxmin < 1e-3)
        self.assertTrue(np.abs(ga_opt.xmin).sum() < 1e-3)

if __name__ == '__main__':
    unittest.main()

