# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
import obsoper
from obsoper import slerp


class TestBilinear(unittest.TestCase):
    def setUp(self):
        self.corners = [obsoper.cartesian(-1, -1),
                        obsoper.cartesian(+1, -1),
                        obsoper.cartesian(+1, +1),
                        obsoper.cartesian(-1, +1)]
        self.centers = [
            slerp.great_circle_nearest_point(self.corners[0],
                                             self.corners[1],
                                             obsoper.cartesian(0, -1)),
            obsoper.cartesian(+1, 0),
            obsoper.cartesian(0, +1),
            obsoper.cartesian(-1, 0)
        ]
        self.values = [1,
                       2,
                       3,
                       4]
        self.center_values = [1.5,
                              2.5,
                              3.5,
                              2.5]

    def test_bilinear_given_lower_left_corner_returns_corner_value(self):
        self.check_bilinear(self.corners[0], self.values[0])

    def test_bilinear_given_lower_right_corner_returns_corner_value(self):
        self.check_bilinear(self.corners[1], self.values[1])

    def test_bilinear_given_upper_left_corner_returns_corner_value(self):
        self.check_bilinear(self.corners[2], self.values[2])

    def test_bilinear_given_upper_right_corner_returns_corner_value(self):
        self.check_bilinear(self.corners[3], self.values[3])

    def test_bilinear_given_bottom_edge_middle(self):
        self.check_bilinear(self.centers[0], self.center_values[0])

    @unittest.skip("too difficult")
    def test_bilinear_given_right_edge_middle(self):
        self.check_bilinear(self.centers[1], self.center_values[1])

    def check_bilinear(self, given, expect):
        result = slerp.bilinear(self.corners, self.values, given)
        self.assertAlmostEqual(expect, result)


class TestLinear(unittest.TestCase):
    def setUp(self):
        # Vertices
        self.first_vertex = (1, 0, 0)
        self.quarterway_vertex = obsoper.cartesian(22.5, 0)
        self.halfway_vertex = obsoper.cartesian(45, 0)
        self.second_vertex = (0, 1, 0)

        # Values
        self.first_value = 1
        self.quarterway_value = 1.25
        self.halfway_value = 1.5
        self.second_value = 2

    def test_linear_given_first_vertex_returns_first_value(self):
        self.check_linear(self.first_vertex, self.first_value)

    def test_linear_given_second_vertex_returns_second_value(self):
        self.check_linear(self.second_vertex, self.second_value)

    def test_linear_given_halfway_vertex_returns_halfway_value(self):
        self.check_linear(self.halfway_vertex, self.halfway_value)

    def test_linear_given_quarterway_vertex_returns_quarterway_value(self):
        self.check_linear(self.quarterway_vertex, self.quarterway_value)

    def check_linear(self, given, expect):
        result = slerp.linear(self.first_vertex,
                              self.second_vertex,
                              self.first_value,
                              self.second_value,
                              given)
        self.assertEqual(expect, result)


class TestGreatCircleNearestPoint(unittest.TestCase):
    def setUp(self):
        # Unit vectors X and Y define great circle one quarter of equator
        self.vertex_a = (1, 0, 0)
        self.vertex_b = (0, 1, 0)

        self.vertex_c = obsoper.cartesian(45, 45)
        self.vertex_t = obsoper.cartesian(45, 0)

    def test_great_circle_nearest_point_given_point_off_great_circle(self):
        result = slerp.great_circle_nearest_point(self.vertex_a,
                                                  self.vertex_b,
                                                  self.vertex_c)
        expect = self.vertex_t
        np.testing.assert_array_almost_equal(expect, result)

    def test_great_circle_nearest_point_given_point_on_great_circle(self):
        result = slerp.great_circle_nearest_point(self.vertex_a,
                                                  self.vertex_b,
                                                  self.vertex_t)
        expect = self.vertex_t
        np.testing.assert_array_almost_equal(expect, result)
