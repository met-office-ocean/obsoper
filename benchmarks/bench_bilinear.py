# pylint: disable=missing-docstring, invalid-name
import numpy as np
import bench
import obsoper.bilinear


class BenchmarkUnitSquare(bench.Suite):
    def setUp(self):
        self.corners = np.array([(0, 0),
                                 (1, 0),
                                 (1, 1),
                                 (0, 1)], dtype="d")
        self.values = np.array([1, 2, 3, 4])

        # __call__ data
        self.fixture = obsoper.bilinear.BilinearTransform(self.corners,
                                                          self.values)
        self.x = np.array([0.1, 0.9], dtype="d")
        self.y = np.array([0.1, 0.9], dtype="d")

        # Positive quadratic root data
        self.a = 0
        self.b = 1
        self.c = -self.y

    def bench_bilinear_transform_init(self):
        for _ in range(100000):
            obsoper.bilinear.BilinearTransform(self.corners,
                                               self.values)

    def bench_bilinear_transform_call(self):
        for _ in range(100000):
            self.fixture(self.x, self.y)

    def bench_bilinear_transform_to_unit_square(self):
        for _ in range(100000):
            self.fixture.to_unit_square(self.x, self.y)

    def bench_bilinear_transform_positive_quadratic_root(self):
        for _ in range(100000):
            self.fixture.positive_quadratic_root(self.a, self.b, self.c)


class BenchmarkRotatedCell(bench.Suite):
    def setUp(self):
        self.corners = np.array([(0.5, 0),
                                 (1, 0.5),
                                 (0.5, 1),
                                 (0, 0.5)], dtype="d")
        self.values = np.array([1, 2, 3, 4])

        # __call__ data
        self.fixture = obsoper.bilinear.BilinearTransform(self.corners,
                                                          self.values)
        self.x = np.array([0.5, 0.1], dtype="d")
        self.y = np.array([0.5, 0.9], dtype="d")

        # Positive quadratic root data
        self.a = -0.5
        self.b = 1.25 - self.x
        self.c = -0.25 + 0.5 * self.x - 0.5 * self.y

    def bench_bilinear_transform_init(self):
        for _ in range(100000):
            obsoper.bilinear.BilinearTransform(self.corners,
                                               self.values)

    def bench_bilinear_transform_call(self):
        for _ in range(100000):
            self.fixture(self.x, self.y)

    def bench_bilinear_transform_to_unit_square(self):
        for _ in range(100000):
            self.fixture.to_unit_square(self.x, self.y)

    def bench_bilinear_transform_positive_quadratic_root(self):
        for _ in range(100000):
            self.fixture.positive_quadratic_root(self.a, self.b, self.c)
