# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
import obsoper


class TestPointInPolygon(unittest.TestCase):
    def setUp(self):
        self.octagon = np.array([(-0.5, -1.0),
                                 (+0.5, -1.0),
                                 (+1.0, -0.5),
                                 (+1.0, +0.5),
                                 (+0.5, +1.0),
                                 (-0.5, +1.0),
                                 (-1.0, +0.5),
                                 (-1.0, -0.5)], dtype="d")

    def test_point_in_polygon_given_octagon_center_returns_true(self):
        self.check_point_in_octagon(0, 0, True)

    def test_point_in_polygon_given_each_vertex_returns_true(self):
        for x, y in self.octagon:
            self.check_point_in_octagon(x, y, True)

    def test_point_in_polygon_bottom_left_corner_returns_false(self):
        self.check_point_in_octagon(-0.8, -0.8, False)

    def check_point_in_octagon(self, x, y, expect):
        result = obsoper.point_in_polygon(self.octagon, x, y)
        self.assertEqual(expect, result)
