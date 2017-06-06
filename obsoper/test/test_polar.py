# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import polar


class TestPolarSearch(unittest.TestCase):
    def test_polar_search(self):
        polar.search()

    def test_side_lengths_given_unit_segment(self):
        points = np.array([[0, 0], [1, 0]])
        result = polar.side_lengths(points)
        expect = [1, 1]
        np.testing.assert_array_almost_equal(expect, result)

    def test_side_lengths_given_triangle(self):
        points = np.array([[0, 0], [1, 0], [1, 1]])
        result = polar.side_lengths(points)
        expect = [1, 1, np.sqrt(2)]
        np.testing.assert_array_almost_equal(expect, result)
