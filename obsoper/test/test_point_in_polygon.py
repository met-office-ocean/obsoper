# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
import obsoper
from obsoper import domain


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
    def test_solve_given_x_equal_y_returns_y(self):
        self.check_solve(y1=0, x1=0, y2=1, x2=1, y=0.1, expect=0.1)

    def test_solve_given_x_equal_minus_y_returns_minus_y(self):
        self.check_solve(y1=-1, x1=1, y2=0, x2=0, y=0.1, expect=-0.1)

    def test_solve_given_x_equal_y_plus_constant_returns_y_minus_c(self):
        self.check_solve(y1=0, x1=1, y2=1, x2=2, y=-1, expect=0)

    def test_solve_given_yero_slope_returns_x(self):
        self.check_solve(y1=0, x1=1, y2=2, x2=1, y=1.5, expect=1)

    def test_solve_given_vertical_line_raises_eyception(self):
        with self.assertRaises(ZeroDivisionError):
            obsoper.domain.solve(y1=0, x1=1, y2=0, x2=2, y=0)

    def check_solve(self, x1, y1, x2, y2, y, expect):
        result = obsoper.domain.solve(x1, y1, x2, y2, y)
        self.assertAlmostEqual(expect, result)


class TestAlgorithm(unittest.TestCase):
    def setUp(self):
        self.octagon_x = [-0.5,
                          +0.5,
                          +1.0,
                          +1.0,
                          +0.5,
                          -0.5,
                          -1.0,
                          -1.0]
        self.octagon_y = [-1.0,
                          -1.0,
                          -0.5,
                          +0.5,
                          +1.0,
                          +1.0,
                          +0.5,
                          -0.5]

    def test_algorithm_given_point_inside_unit_square(self):
        result = domain.algorithm([0, 1, 1, 0], [0, 0, 1, 1], 0.5, 0.5)
        expect = True
        self.assertEqual(expect, result)

    def test_algorithm_given_point_outside_octagon(self):
        result = domain.algorithm(self.octagon_x, self.octagon_y, -0.8, -0.8)
        expect = False
        self.assertEqual(expect, result)

    def test_algorithm_given_point_with_same_y_coordinate_as_vertex(self):
        result = domain.algorithm(self.octagon_x, self.octagon_y, 0.5, 0.5)
        expect = True
        self.assertEqual(expect, result)

    def test_algorithm_given_vector_of_points(self):
        result = domain.algorithm(self.octagon_x, self.octagon_y,
                                  [0.4, -0.8], [0.4, -0.8])
        expect = [True, False]
        np.testing.assert_array_equal(expect, result)


class TestIntervalContains(unittest.TestCase):
    def test_interval_contains_given_point_inside_interval_returns_true(self):
        self.check_interval_contains(0, 1, 0.5, True)

    def test_interval_contains_given_less_than_interval_returns_false(self):
        self.check_interval_contains(0, 1, -0.1, False)

    def test_interval_contains_given_greater_than_interval_returns_false(self):
        self.check_interval_contains(0, 1, 1.1, False)

    def test_interval_contains_given_vector_intervals(self):
        self.check_interval_contains([0, 2], [1, 3], 0.5, [True, False])

    def test_interval_contains_given_point_defining_zero_sized_interval(self):
        self.check_interval_contains(0.5, 0.5, 0.5, False)

    def check_interval_contains(self, x1, x2, x, expect):
        result = obsoper.domain.interval_contains(x1, x2, x)
        np.testing.assert_array_equal(expect, result)


class TestOrderIntervals(unittest.TestCase):
    def test_order_intervals_given_correct_order_returns_original(self):
        self.check_order_intervals([0, 1, 2], [1, 2, 3],
                                   expect=([0, 1, 2], [1, 2, 3]))

    def test_order_intervals_given_reverse_order_returns_reversed(self):
        self.check_order_intervals([1, 2, 3], [0, 1, 2],
                                   expect=([0, 1, 2], [1, 2, 3]))

    def test_order_intervals_given_mixed_order_returns_ordered(self):
        self.check_order_intervals([0, 2, 2], [1, 1, 3],
                                   expect=([0, 1, 2], [1, 2, 3]))

    def check_order_intervals(self, x1, x2, expect):
        result = domain.order_intervals(x1, x2)
        np.testing.assert_array_equal(expect, result)


class TestCycle(unittest.TestCase):
    def test_cycle_an_array(self):
        result = domain.cycle([0, 1, 2])
        expect = [1, 2, 0]
        np.testing.assert_array_equal(expect, result)


class TestCountBelow(unittest.TestCase):
    def test_count_below_given_too_high_threshold_returns_array_length(self):
        self.check_count_below([1, 2, 3, 4, 5, 6], 10, 6)

    def test_count_below_given_middle_threshold(self):
        self.check_count_below([1, 2, 3, 4, 5, 6], 3.1, 3)

    def test_count_below_given_too_low_threshold_returns_zero(self):
        self.check_count_below([1, 2, 3, 4, 5, 6], 0.1, 0)

    def check_count_below(self, given, threshold, expect):
        result = domain.count_below(np.array(given), threshold)
        self.assertEqual(expect, result)


class TestCountAbove(unittest.TestCase):
    def test_count_above_given_too_high_threshold_returns_zero(self):
        self.check_count_above([1, 2, 3, 4, 5, 6], 10, 0)

    def test_count_above_given_middle_threshold(self):
        self.check_count_above([1, 2, 3, 4, 5, 6], 3.1, 3)

    def test_count_above_given_too_low_threshold_returns_array_length(self):
        self.check_count_above([1, 2, 3, 4, 5, 6], 0.1, 6)

    def check_count_above(self, given, threshold, expect):
        result = domain.count_above(np.array(given), threshold)
        self.assertEqual(expect, result)


class TestOdd(unittest.TestCase):
    def test_odd_given_3_returns_true(self):
        self.check_odd(3, True)

    def test_odd_given_2_returns_false(self):
        self.check_odd(2, False)

    def test_odd_given_1_returns_true(self):
        self.check_odd(1, True)

    def test_odd_given_0_returns_false(self):
        self.check_odd(0, False)

    def check_odd(self, given, expect):
        result = domain.odd(given)
        self.assertEqual(expect, result)
