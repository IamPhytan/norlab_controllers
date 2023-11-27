import time
import unittest

import numpy as np
from norlabcontrollib.path import Path, VectorizedPath

# import matplotlib.pyplot as plt


class PathTest(unittest.TestCase):
    def __init__(self, methodName: str = "runTest", n: int = 10) -> None:
        super().__init__(methodName)
        self.n = n

    test_path_poses = np.load("tests/traj_a_int.npy")

    def setUp(self):
        reference_path = Path(self.test_path_poses)
        reference_path.compute_curvatures()
        self.curvatures = reference_path.curvatures

        self.original_path = Path(self.test_path_poses)
        self.vectorized_path = VectorizedPath(self.test_path_poses)

        self.start_time = time.perf_counter()

    def tearDown(self):
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        print(f"{self.id()} : {duration/self.n:0.4f} s")

    def testPathComputeCurvatures(self):
        for _ in range(self.n):
            self.original_path.compute_curvatures()
        np.testing.assert_allclose(self.original_path.curvatures, self.curvatures)

    def testVectorizedPathComputeCurvatures(self):
        for _ in range(self.n):
            self.vectorized_path.compute_curvatures()
        np.testing.assert_allclose(self.vectorized_path.curvatures, self.curvatures)


# if __name__ == "__main__":
#     test_path_poses = np.load("tests/traj_a_int.npy")

#     test_common_path = Path(test_path_poses)
#     common_time = time_function(test_common_path.compute_curvatures)

#     test_vector_path = VectorizedPath(test_path_poses)
#     vector_time = time_function(test_vector_path.compute_curvatures)

#     print(f"Original time : {common_time} s")
#     print(f"Vectorized time : {vector_time} s")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(PathTest)
    unittest.TextTestRunner(verbosity=0).run(suite)
