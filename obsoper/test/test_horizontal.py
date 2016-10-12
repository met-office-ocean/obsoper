# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper.horizontal import UnitSquare


class TestUnitSquare(unittest.TestCase):
    def setUp(self):
        self.empty = []
        self.data = np.array([1, 2, 3, 4]).reshape(2, 2)

    def test_weights_given_lower_left_corner(self):
        self.check_weights([0], [0], [1, 0, 0, 0])

    def test_weights_given_lower_right_corner(self):
        self.check_weights([0], [1], [0, 1, 0, 0])

    def test_weights_given_upper_left_corner(self):
        self.check_weights([1], [0], [0, 0, 1, 0])

    def test_weights_given_upper_right_corner(self):
        self.check_weights([1], [1], [0, 0, 0, 1])

    def check_weights(self, x, y, expect):
        fixture = UnitSquare(self.empty, self.empty,
                             np.array(x), np.array(y))
        result = fixture.weights
        expect = np.array(expect).reshape(4, 1)
        np.testing.assert_array_equal(expect, result)

    def test_weights_given_n_fractions_returns_4_by_n_shape(self):
        x, y = np.array([0, 1]), np.array([0, 1])
        fixture = UnitSquare(self.empty, self.empty, x, y)
        result = fixture.weights.shape
        expect = (4, 2)
        self.assertEqual(expect, result)

    def test_values_given_index_and_data(self):
        i, j = np.array([0]), np.array([0])
        fixture = UnitSquare(i, j, self.empty, self.empty)
        data = np.array([1, 2, 3, 4]).reshape(2, 2)
        result = fixture.values(data)
        expect = np.array([1, 2, 3, 4]).reshape(4, 1)
        np.testing.assert_array_equal(expect, result)

    def test_interpolation_given_lower_left_corner(self):
        self.check_interpolation([0], [0], [1])

    def test_interpolation_given_lower_right_corner(self):
        self.check_interpolation([0], [1], [2])

    def test_interpolation_given_upper_left_corner(self):
        self.check_interpolation([1], [0], [3])

    def test_interpolation_given_upper_right_corner(self):
        self.check_interpolation([1], [1], [4])

    @staticmethod
    def check_interpolation(x, y, expect):
        x, y = np.array(x), np.array(y)
        i, j = np.array([0]), np.array([0])
        fixture = UnitSquare(i, j, x, y)
        data = np.array([1, 2, 3, 4]).reshape(2, 2)
        result = fixture(data)
        np.testing.assert_array_equal(expect, result)

    def test_masked_given_upper_left_masked_returns_true(self):
        i, j = np.array([0]), np.array([0])
        fixture = UnitSquare(i, j, self.empty, self.empty)
        data = np.ma.masked_array([[1, 2], [3, 4]])
        data[1, 0] = np.ma.masked
        result = fixture.masked(data)
        expect = np.array([True])
        np.testing.assert_array_equal(expect, result)

    def test_interpolation_given_3d_data(self):
        data = np.zeros((3, 3, 2))
        data[..., 0] = 1
        data[..., 1] = 2
        x, y = np.array([0]), np.array([0])
        i, j = np.array([1]), np.array([1])
        fixture = UnitSquare(i, j, x, y)
        result = fixture(data)
        expect = np.array([1, 2]).reshape(1, 2)
        np.testing.assert_array_equal(expect, result)
