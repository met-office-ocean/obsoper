# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper.exceptions import StepNotFound
from obsoper import (cell,
                     cursors,
                     walk)


class TestWalk(unittest.TestCase):
    def setUp(self):
        longitudes, latitudes = np.meshgrid([100, 104, 106, 107],
                                            [80, 83, 84],
                                            indexing="ij")
        self.fixture = walk.Walk.from_lonlats(longitudes,
                                              latitudes)

    def test_query_given_multiple_locations_and_starting_indices(self):
        longitudes, latitudes = np.array([100, 106.1]), np.array([80, 83])
        i, j = np.array([1, 1]), np.array([1, 1])
        result = self.fixture.query(longitudes, latitudes, i, j)
        expect = np.array([0, 2]), np.array([0, 1])
        np.testing.assert_array_equal(expect, result)

    def test_grid_walk_given_point_in_maximum_grid_cell(self):
        self.check_walk((106.9, 83.9), i=0, j=0, expect=(2, 1))

    def test_grid_walk_given_point_one_grid_cell_north(self):
        self.check_walk((100.1, 83.9), i=0, j=0, expect=(0, 1))

    def test_grid_walk_given_point_two_grid_cells_east(self):
        self.check_walk((106.9, 80), i=0, j=0, expect=(2, 0))

    def test_grid_walk_starting_at_north_east_corner_travelling_south(self):
        self.check_walk((100, 80), i=2, j=1, expect=(0, 0))

    def test_grid_walk_given_i_maximum_performs_search(self):
        self.check_walk((100, 80), i=3, j=0, expect=(0, 0))

    def test_grid_walk_given_j_maximum_performs_search(self):
        self.check_walk((100, 80), i=0, j=2, expect=(0, 0))

    def check_walk(self, point, i, j, expect):
        result = self.fixture.query_one(point, i, j)
        np.testing.assert_array_almost_equal(expect, result)

    def test_walk_across_northfold(self):
        # North fold along 100th meridian
        lon0, dlon = 100, 0.5
        lat0, lat1, lat2 = 80, 81, 82

        # Neighbouring grid boxes either side of fold
        longitudes = np.array([[lon0 + dlon, lon0],
                               [lon0 + dlon, lon0],
                               [lon0 - dlon, lon0],
                               [lon0 - dlon, lon0],
                               [lon0 - dlon, lon0]])
        latitudes = np.array([[lat1, lat1],
                              [lat0, lat0],
                              [lat0, lat0],
                              [lat1, lat1],
                              [lat2, lat2]])

        # Point in center of second grid box
        point = (lon0 - dlon / 2., (lat1 + lat0) / 2.)

        fixture = walk.Walk.tripolar(longitudes, latitudes, fold_index=1)

        result = fixture.query_one(point, i=0, j=0)
        expect = (2, 0)
        self.assertEqual(expect, result)

    def test_walk_across_dateline(self):
        """Stepping algorithms should walk in direction of point"""
        longitudes, latitudes = np.meshgrid([178, 179, -179, -178],
                                            [10, 20, 30, 40],
                                            indexing="ij")
        fixture = walk.Walk.from_lonlats(longitudes,
                                         latitudes)
        point = (-178.5, 25)
        fixture = walk.Walk.from_lonlats(longitudes, latitudes)
        result = fixture.query_one(point, i=0, j=1)
        expect = (2, 1)
        self.assertEqual(expect, result)

    def test_diagonal_stepping_edge_case(self):
        """Extremely rare edge case consisting of diagonal neighbours
        neither of which contain the point. And whose displacement vectors
        produce (di, dj) pairs both of which are greater than 1.

        .. note:: case taken directly from orca grid and real observation
        """
        longitudes = np.array([[82.96250916, 78.00720978, 73.],
                               [83.98399353, 78.52962494, 73.],
                               [85.23661041, 79.1739502, 73.]])
        latitudes = np.array([[88.83627319, 88.84931946, 88.85367584],
                              [88.94418335, 88.95856476, 88.96337891],
                              [89.05162811, 89.06764984, 89.07302094]])
        point = (78.830612, 88.958031)

        fixture = walk.Walk.from_lonlats(longitudes, latitudes)
        result = fixture.query_one(point, i=0, j=0)
        expect = (1, 0)
        self.assertEqual(expect, result)

    def test_point_above_line_edge_case(self):
        """Simple edge case taken from real data that produces an infinite loop

        .. note:: case taken directly from orca grid and real observation
        """
        longitudes = np.array([[-122.912, -121.936, -120.955, -119.968],
                               [-122.601, -121.643, -120.679, -119.711],
                               [-122.301, -121.360, -120.413, -119.462],
                               [-122.011, -121.086, -120.156, -119.222]])
        latitudes = np.array([[84.358, 84.387, 84.414, 84.439],
                              [84.258, 84.286, 84.312, 84.337],
                              [84.158, 84.185, 84.211, 84.235],
                              [84.057, 84.085, 84.110, 84.134]])
        point = (-121.27652, 84.300476)

        fixture = walk.Walk.from_lonlats(longitudes, latitudes)
        result = fixture.query_one(point, i=2, j=1)
        expect = (0, 1)
        self.assertEqual(expect, result)

    def test_point_inside_great_circle_and_lonlat_line(self):
        """Great circle segments are not straight lines in longitude/latitude
        space.

        .. note:: Case taken directly from orca grid and real observation.
                  The effect is sensitive to the precision of the numbers.
        """
        longitudes = np.array([[-125.14778137, -123.9805603],
                               [-124.73101807, -123.58721924],
                               [-124.33137512, -123.21022034]])
        latitudes = np.array([[85.28964233, 85.32194519],
                              [85.18976593, 85.22135925],
                              [85.08974457, 85.12065887]])
        point = (-124.52118, 85.195602)

        fixture = walk.Walk.from_lonlats(longitudes, latitudes)
        result = fixture.query_one(point, i=2, j=0)
        expect = (2, 0)
        self.assertEqual(expect, result)


class TestWalkStep(unittest.TestCase):
    def setUp(self):
        ni, nj = 4, 3
        longitudes, latitudes = np.meshgrid([100, 104, 106, 107],
                                            [80, 83, 84],
                                            indexing="ij")
        self.cells = cell.Collection.from_lonlats(longitudes, latitudes)
        self.cursor = cursors.Cursor(ni, nj)
        self.fixture = walk.Walk(self.cells, self.cursor)

    def test_query_one_given_point_in_cell_0_0(self):
        self.check_query_one((100, 80), (0, 0))

    def test_query_one_given_point_in_cell_1_0(self):
        self.check_query_one((104.1, 80), (1, 0))

    def test_query_one_given_point_in_cell_0_1(self):
        self.check_query_one((100, 83.1), (0, 1))

    def test_query_one_given_point_in_northernmost_cell(self):
        self.check_query_one((100, 84), (0, 1))

    def test_query_one_given_point_in_easternmost_cell(self):
        self.check_query_one((107, 80), (2, 0))

    def check_query_one(self, point, expect):
        result = self.fixture.query_one(point)
        self.assertEqual(expect, result)

    def test_direction_given_east_point(self):
        self.check_direction(0, 0, (107, 80), (1, 0))

    def test_direction_given_north_point(self):
        self.check_direction(0, 0, (100, 84), (0, 1))

    def test_direction_given_west_point(self):
        self.check_direction(1, 0, (100, 80), (-1, 0))

    def test_direction_given_south_point(self):
        self.check_direction(1, 1, (104, 80), (0, -1))

    def check_direction(self, i, j, point, expect):
        result = self.fixture.direction(self.cells[i, j], point)
        self.assertEqual(expect, result)

    def test_direction_given_parallelogram_cell_intersecting_0E(self):
        longitudes = np.array([[-0.12521362, 0.27581787],
                               [0.95932007, 1.37954712]])
        latitudes = np.array([[84.45360565, 84.56639099],
                              [84.41712189, 84.52923584]])
        i, j = 0, 0
        point = (0.90654248, 84.620514)

        cells = cell.Collection.from_lonlats(longitudes, latitudes)
        cursor = cursors.Cursor(ni=2, nj=2)
        fixture = walk.Walk(cells, cursor)
        result = fixture.direction(cells[i, j], point)
        expect = (0, 1)
        self.assertEqual(expect, result)


class TestNextStep(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([(0., 0.),
                                  (1., 0.),
                                  (1., 1.),
                                  (0., 1.)],
                                 dtype=np.double)
        self.center = (0.5, 0.5)

        self.due_north = (0.1, 1.1)
        self.due_south = (0.1, -0.1)
        self.due_east = (1.1, 0.1)
        self.due_west = (-0.1, 0.1)

        self.north = (0, 1)
        self.south = (0, -1)
        self.east = (1, 0)
        self.west = (-1, 0)

        self.inside_arc = (0.5, 1.00001)
        self.inside_box = (0.5, 0.5)

    def test_next_step_given_point_north_returns_north(self):
        self.check_next_step(self.due_north, self.north)

    def test_next_step_given_point_south_returns_south(self):
        self.check_next_step(self.due_south, self.south)

    def test_next_step_given_point_east_returns_east(self):
        self.check_next_step(self.due_east, self.east)

    def test_next_step_given_point_west_returns_west(self):
        self.check_next_step(self.due_west, self.west)

    def test_next_step_given_point_inside_great_circle_raises_exception(self):
        with self.assertRaises(StepNotFound):
            walk.next_step(self.vertices, self.center, self.inside_arc)

    def test_next_step_given_point_inside_box_raises_exception(self):
        with self.assertRaises(StepNotFound):
            walk.next_step(self.vertices, self.center, self.inside_box)

    def check_next_step(self, position, expect):
        result = walk.next_step(self.vertices, self.center, position)
        self.assertEqual(expect, result)
