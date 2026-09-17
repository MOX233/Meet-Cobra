import unittest
import numpy as np
from utils.alg_utils import km_algorithm
from utils.compiled_matching import km_algorithm_compiled


class CompiledMatchingTest(unittest.TestCase):
    def test_exact_matching_and_tie_order(self):
        rng = np.random.default_rng(14)
        for rows, columns in ((1,1), (4,9), (9,4), (25,25), (50,55), (128,132)):
            for cost in (rng.uniform(-1,1,(rows,columns)),
                         rng.integers(-1,3,(rows,columns)).astype(float),
                         np.zeros((rows,columns))):
                original = km_algorithm(cost)
                compiled = km_algorithm_compiled(cost)
                self.assertEqual(original[0], compiled[0])
                self.assertEqual(original[1], compiled[1])


if __name__ == "__main__":
    unittest.main()
