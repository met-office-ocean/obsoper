# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import orca


class TestNorthFold(unittest.TestCase):
    def test_north_fold_given_point_returns_point_on_opposite_side(self):
        result = orca.north_fold([0, 0], [0, 0])
        expect = {0: 1,
                  1: 0}
        self.assertEqual(expect, result)

    def test_north_fold_given_multiple_pairs_of_points(self):
        result = orca.north_fold(longitudes=[0, 1, 2, 2, 1, 0],
                                 latitudes=[1, 2, 0, 0, 2, 1])
        expect = {0: 5,
                  1: 4,
                  2: 3,
                  3: 2,
                  4: 1,
                  5: 0}
        self.assertEqual(expect, result)

    def test_north_fold_given_empty_arrays_returns_empty_dict(self):
        result = orca.north_fold(np.array([]), np.array([]))
        expect = {}
        self.assertEqual(expect, result)


class TestRemoveHalo(unittest.TestCase):
    def test_remove_halo_given_empty_array_returns_empty_array(self):
        self.check_remove_halo([], [])

    def test_remove_halo_given_2_by_2_returns_empty_array(self):
        self.check_remove_halo([[1, 1], [1, 1]], [])

    def test_remove_halo_given_3_by_3_returns_1_by_2_array(self):
        self.check_remove_halo([[1, 1, 1],
                                [0, 0, 1],
                                [1, 1, 1]], [[0, 0]])

    def check_remove_halo(self, given, expect):
        result = orca.remove_halo(given)
        np.testing.assert_array_equal(expect, result)
