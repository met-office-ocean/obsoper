# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import spherical


class TestIntersect(unittest.TestCase):
    def test_intersect_given_intersecting_segments_returns_true(self):
        equator = [(-10, 0), (10, 0)]
        greenwich = [(0, -10), (0, 10)]
        self.check_intersect(equator, greenwich, True)

    def test_intersect_given_non_intersecting_segments_returns_false(self):
        equator = [(-10, 0), (10, 0)]
        line_2 = [(20, -10), (20, 10)]
        self.check_intersect(equator, line_2, False)

    def test_intersect_given_segments_parallel_to_equator_returns_false(self):
        self.check_intersect([(0, 10), (100, 10)],
                             [(0, -10), (100, -10)],
                             False)

    def test_intersect_given_diagonal_arcs(self):
        self.check_intersect([(104, 80), (106, 83)],
                             [(106, 80), (104, 83)],
                             True)

    def check_intersect(self, line_1, line_2, expect):
        result = spherical.intersect(np.asarray(line_1, dtype="d"),
                                     np.asarray(line_2, dtype="d"))
        self.assertEqual(expect, result)


class TestIntercept(unittest.TestCase):
    def test_intercept_given_intersecting_segments_returns_point(self):
        equator = np.array([(-10, 0), (10, 0)], dtype="d")
        greenwich = np.array([(0, -10), (0, 10)], dtype="d")
        result = spherical.intercept(equator,
                                     greenwich)
        expect = (0, 0)
        np.testing.assert_array_almost_equal(expect, result)

    def test_intersect_given_non_intersecting_segments_raises_exception(self):
        line_1 = np.array([(-10, 0), (10, 0)], dtype="d")
        line_2 = np.array([(20, -10), (20, 10)], dtype="d")
        with self.assertRaises(Exception):
            spherical.intercept(line_1, line_2)


class TestToCartesian(unittest.TestCase):
    def test_to_cartesian_given_equator_returns_1_0_0(self):
        self.check_to_cartesian([0, 0], (1, 0, 0))

    def test_to_cartesian_given_90th_meridian_returns_0_1_0(self):
        self.check_to_cartesian([90, 0], (0, 1, 0))

    def test_to_cartesian_given_north_pole_returns_0_0_1(self):
        self.check_to_cartesian([0, 90], (0, 0, 1))

    @staticmethod
    def check_to_cartesian(point, expect):
        result = spherical.to_cartesian(point)
        np.testing.assert_array_almost_equal(expect, result)


class TestToSpherical(unittest.TestCase):
    def test_to_spherical_given_equator_returns_1_0_0(self):
        self.check_to_spherical((1, 0, 0), (0, 0))

    def test_to_spherical_given_90th_meridian_returns_0_1_0(self):
        self.check_to_spherical((0, 1, 0), (90, 0))

    def test_to_spherical_given_north_pole_returns_0_0_1(self):
        self.check_to_spherical((0, 0, 1), (0, 90))

    @staticmethod
    def check_to_spherical(point, expect):
        result = spherical.to_spherical(point)
        np.testing.assert_array_almost_equal(expect, result)
