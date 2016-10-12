# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import box


class TestBox(unittest.TestCase):
    def setUp(self):
        polygon = np.array([(0.5, 0.0),
                            (1.0, 0.5),
                            (0.5, 1.0),
                            (0.0, 0.5)],
                           dtype="d")
        self.fixture = box.Box.frompolygon(polygon)

        self.outside_north = (0.5, 1.1)
        self.outside_south = (0.5, -0.1)
        self.outside_east = (1.1, 0.5)
        self.outside_west = (-0.1, 0.5)

        self.north_edge = (0.1, 1.0)
        self.south_edge = (0.1, 0.0)
        self.east_edge = (1.0, 0.1)
        self.west_edge = (0.0, 0.1)

        self.point_inside = (0.1, 0.1)

    def test_inside_given_outside_north_returns_false(self):
        self.check_inside(self.outside_north, False)

    def test_inside_given_outside_south_returns_false(self):
        self.check_inside(self.outside_south, False)

    def test_inside_given_outside_east_returns_false(self):
        self.check_inside(self.outside_east, False)

    def test_inside_given_outside_west_returns_false(self):
        self.check_inside(self.outside_west, False)

    def test_inside_given_point_on_north_edge_returns_true(self):
        self.check_inside(self.north_edge, True)

    def test_inside_given_point_on_south_edge_returns_true(self):
        self.check_inside(self.south_edge, True)

    def test_inside_given_point_on_east_edge_returns_true(self):
        self.check_inside(self.east_edge, True)

    def test_inside_given_point_on_west_edge_returns_true(self):
        self.check_inside(self.west_edge, True)

    def test_inside_given_point_inside_returns_true(self):
        self.check_inside(self.point_inside, True)

    def check_inside(self, position, expect):
        result = self.fixture.inside(position[0], position[1])
        self.assertEqual(expect, result)


class TestDateline(unittest.TestCase):
    def setUp(self):
        polygon = np.array([(+179, 0),
                            (-179, 0),
                            (-179, 1),
                            (+179, 1)],
                           dtype="d")
        self.square = box.Dateline(polygon)

        polygon = np.array([(+178, 0),
                            (-179, 0),
                            (-178, 1),
                            (+179, 1)],
                           dtype="d")
        self.trapezoid = box.Dateline(polygon)

    def test_inside_given_point_west_of_180_inside_cell_returns_true(self):
        self.check_inside(self.square, +179.5, 0.5, True)

    def test_inside_given_point_north_of_cell_returns_false(self):
        self.check_inside(self.square, +179.5, 2.0, False)

    def test_inside_x_given_point_west_of_180_inside_box_returns_true(self):
        self.check_inside_x(self.square, +179.5, True)

    def test_inside_x_given_point_east_of_180_inside_box_returns_true(self):
        self.check_inside_x(self.square, -179.5, True)

    def test_inside_x_given_point_west_of_180_outside_box_returns_false(self):
        self.check_inside_x(self.square, +170.0, False)

    def test_inside_x_given_point_east_of_180_outside_box_returns_false(self):
        self.check_inside_x(self.square, -170.0, False)

    def test_inside_y_given_point_north_of_cell_returns_false(self):
        self.check_inside_y(self.square, 2.0, False)

    def test_inside_y_given_point_between_north_south_returns_true(self):
        self.check_inside_y(self.square, 0.5, True)

    def test_inside_y_given_point_south_of_cell_returns_false(self):
        self.check_inside_y(self.square, -2.0, False)

    def test_trapezoidal_cell_inside_x_given_point_inside_east_triangle(self):
        """important check longitude bounds of dateline cell are correct"""
        self.check_inside_x(self.trapezoid, -178.5, True)

    def test_trapezoidal_cell_inside_x_given_point_inside_west_triangle(self):
        """important check longitude bounds of dateline cell are correct"""
        self.check_inside_x(self.trapezoid, +178.5, True)

    def check_inside(self, cell, longitude, latitude, expect):
        result = cell.inside(longitude, latitude)
        self.assertEqual(expect, result)

    def check_inside_x(self, cell, coordinate, expect):
        result = cell.inside_x(coordinate)
        self.assertEqual(expect, result)

    def check_inside_y(self, cell, coordinate, expect):
        result = cell.inside_y(coordinate)
        self.assertEqual(expect, result)
