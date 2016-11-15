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
        self.concave = np.array([(0, 0),
                                 (1, 0),
                                 (1, 1),
                                 (2, 1),
                                 (2, 0),
                                 (3, 0),
                                 (3, 3),
                                 (0, 3)], dtype="d")

    def test_point_in_polygon_given_octagon_center_returns_true(self):
        self.check_point_in_polygon(self.octagon, 0, 0, True)

    def test_point_in_polygon_given_each_vertex_returns_true(self):
        for x, y in self.octagon:
            print x, y
            self.check_point_in_polygon(self.octagon, x, y, True)

    def test_point_in_polygon_bottom_left_corner_returns_false(self):
        self.check_point_in_polygon(self.octagon, -0.8, -0.8, False)

    def test_point_in_polygon_concave_outside_returns_false(self):
        self.check_point_in_polygon(self.concave, 1.5, 0.5, False)

    def test_point_in_polygon_concave_inside_returns_true(self):
        self.check_point_in_polygon(self.concave, 1.5, 1.5, True)

    def check_point_in_polygon(self, vertices, x, y, expect):
        result = obsoper.point_in_polygon(vertices, (x, y))
        self.assertEqual(expect, result)


class TestBoundary(unittest.TestCase):
    def test_boundary(self):
        longitudes, latitudes = np.meshgrid([0, 1, 2],
                                            [0, 1, 2],
                                            indexing="ij")
        result = obsoper.domain.boundary(longitudes,
                                         latitudes)
        expect = [(0, 0),
                  (1, 0),
                  (2, 0),
                  (2, 1),
                  (2, 2),
                  (1, 2),
                  (0, 2),
                  (0, 1)]
        np.testing.assert_array_equal(expect, result)


class TestSolve(unittest.TestCase):
    def test_solve_given_y_equal_x_returns_x(self):
        self.check_solve(x1=0, y1=0, x2=1, y2=1, x=0.1, expect=0.1)

    def test_solve_given_y_equal_minus_x_returns_minus_x(self):
        self.check_solve(x1=-1, y1=1, x2=0, y2=0, x=0.1, expect=-0.1)

    def test_solve_given_y_equal_x_plus_constant_returns_x_minus_c(self):
        self.check_solve(x1=0, y1=1, x2=1, y2=2, x=-1, expect=0)

    def test_solve_given_zero_slope_returns_y(self):
        self.check_solve(x1=0, y1=1, x2=2, y2=1, x=1.5, expect=1)

    def test_solve_given_vertical_line_raises_exception(self):
        with self.assertRaises(ZeroDivisionError):
            obsoper.domain.solve(x1=0, y1=1, x2=0, y2=2, x=0)

    def check_solve(self, x1, y1, x2, y2, x, expect):
        result = obsoper.domain.solve(x1, y1, x2, y2, x)
        self.assertAlmostEqual(expect, result)


class TestIntervalContains(unittest.TestCase):
    def test_interval_contains_given_point_inside_interval_returns_true(self):
        self.check_interval_contains(0, 1, 0.5, True)

    def test_interval_contains_given_less_than_interval_returns_false(self):
        self.check_interval_contains(0, 1, -0.1, False)

    def test_interval_contains_given_greater_than_interval_returns_false(self):
        self.check_interval_contains(0, 1, 1.1, False)

    def check_interval_contains(self, x1, x2, x, expect):
        result = obsoper.domain.interval_contains(x1, x2, x)
        self.assertEqual(expect, result)


class TestCountIntersects(unittest.TestCase):
    def setUp(self):
        self.polygon = [(0, 0),
                        (1, 0),
                        (1, 1),
                        (0, 1)]
        self.ray = [(-0.5, -0.5), (0.5, 0.5)]

    def test_count_intersects_given_ray_through_vertex_returns_one(self):
        result = obsoper.domain.count_intersects(self.polygon, self.ray)
        expect = 1
        self.assertEqual(expect, result)


class TestLineSegmentsIntersect(unittest.TestCase):
    def test_segments_intersect_given_unit_i_unit_j_returns_true(self):
        self.check_segments_intersect([(0, 0), (1, 0)],
                                      [(0, 0), (0, 1)],
                                      True)

    def test_segments_intersect_given_touching_lines_returns_true(self):
        self.check_segments_intersect([(0, 0), (5, 5)],
                                      [(2, 2), (3, 8)],
                                      True)

    def test_segments_intersect_given_negative_touching_lines_returns_true(self):
        self.check_segments_intersect([(-2, -2), (-2, 2)],
                                      [(-2, 0), (0, 0)],
                                      True)

    def test_segments_intersect_given_line_to_right_returns_false(self):
        self.check_segments_intersect([(0, 0), (1, 0)],
                                      [(2, 0), (3, 0)],
                                      False)

    def test_segments_intersect_given_intersecting_lines_returns_true(self):
        self.check_segments_intersect([(0, 0), (1, 0)],
                                      [(0.5, -1), (0.5, 1)],
                                      True)

    def test_segments_intersect_given_line_to_left_returns_false(self):
        self.check_segments_intersect([(0, 0), (1, 0)],
                                      [(-2, 0), (-1, 0)],
                                      False)

    def test_segments_intersect_given_short_line_above_returns_false(self):
        self.check_segments_intersect([(0, 0), (1, 0)],
                                      [(0.1, 1), (1, 1)],
                                      False)

    def test_segments_intersect_given_short_line_below_returns_false(self):
        self.check_segments_intersect([(0, 0), (1, 0)],
                                      [(0.1, -1), (1, -1)],
                                      False)

    def test_segments_intersect_given_interior_bounding_box_no_overlap(self):
        self.check_segments_intersect([(0, 0), (1, 1)],
                                      [(0.5, 0.25), (0.75, 0.5)],
                                      False)

    def test_segments_intersect_given_colinear_disjoint_lines_returns_false(self):
        self.check_segments_intersect([(-2, -2), (4, 4)],
                                      [(6, 6), (10, 10)],
                                      False)

    def test_segments_intersect_given_colinear_interior_line_returns_true(self):
        self.check_segments_intersect([(0, 0), (10, 10)],
                                      [(2, 2), (6, 6)],
                                      True)

    def test_segments_intersect_given_reversed_line_returns_true(self):
        self.check_segments_intersect([(6, 8), (10, -2)],
                                      [(10, -2), (6, 8)],
                                      True)

    def check_segments_intersect(self, line_1, line_2, expect):
        result = obsoper.domain.segments_intersect(line_1, line_2)
        self.assertEqual(expect, result)


class TestSide(unittest.TestCase):
    def setUp(self):
        self.line = [(0, 1), (1, 1)]
        self.right_point = (0, 0)
        self.on_line = (0.5, 1)
        self.left_point = (0, 2)

    def test_side_given_point_on_right(self):
        self.check_side(self.line, self.right_point, -1)

    def test_side_given_point_on_left(self):
        self.check_side(self.line, self.left_point, +1)

    def test_side_given_point_on_line(self):
        self.check_side(self.line, self.on_line, 0)

    def check_side(self, line, point, expect):
        result = obsoper.domain.side(line, point)
        self.assertAlmostEqual(expect, result)
