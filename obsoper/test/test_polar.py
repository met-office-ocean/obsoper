# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from scipy import spatial
from obsoper import polar


class TestPolarSearch(unittest.TestCase):
    def test_polar_search(self):
        polar.search()

    def test_join_simplices(self):
        result = polar.join_simplices([2, 3, 1], [1, 3, 0], 1)
        expect = [2, 3, 0, 1]
        self.assertEqual(expect, result)

    def test_longest_side_given_right_angle_triangle(self):
        points = np.array([[0, 0], [1, 0], [1, 1]])
        result = polar.longest_side(points)
        expect = 2
        self.assertEqual(expect, result)

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


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.unit_square = np.array([[0, 0], [1, 0], [1, 1]])

    def test_grid_structure(self):
        polar.grid_structure()

    def test_grid(self):
        fixture = polar.Grid(self.unit_square)
        result = fixture.cells
        expect = np.array([[0, 1, 2, 3]])
        np.testing.assert_array_equal(expect, result)

    def test_grid_find_cell(self):
        fixture = polar.Grid(self.unit_square)
        result = fixture.find_cell([0.5, 0.5])
        expect = 0
        np.testing.assert_array_equal(expect, result)

    def test_delaunay_given_right_angled_tringle(self):
        points = np.array([[0, 0], [1, 0], [1, 1]])
        fixture = spatial.Delaunay(points)
        result = fixture.simplices
        expect = np.array([[1, 2, 0]])
        np.testing.assert_array_equal(expect, result)

    def test_delaunay_given_unit_square(self):
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        fixture = spatial.Delaunay(points)
        result = fixture.simplices
        expect = np.array([[2, 3, 1],
                           [1, 3, 0]])
        np.testing.assert_array_equal(expect, result)
