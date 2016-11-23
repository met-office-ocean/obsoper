# pylint: disable=missing-docstring, invalid-name
import unittest
import numpy as np
from obsoper import cell


class TestCellContains(unittest.TestCase):
    def setUp(self):
        vertices = [(0, 0),
                    (0, 1),
                    (1, 1),
                    (1, 0)]
        self.square = cell.Cell(np.asarray(vertices, dtype="d"))

        vertices = [(0.5, 0.0),
                    (1.0, 0.5),
                    (0.5, 1.0),
                    (0.0, 0.5)]
        self.diamond = cell.Cell(np.asarray(vertices, dtype="d"))

    def test_contains_given_lower_left_corner_returns_true(self):
        self.check_contains(self.square, 0., 0., True)

    def test_contains_given_square_and_center_returns_true(self):
        self.check_contains(self.square, 0.5, 0.5, True)

    def test_square_contains_given_point_to_left_returns_false(self):
        self.check_contains(self.square, 1.5, 0.5, False)

    def test_square_contains_given_point_to_right_returns_false(self):
        self.check_contains(self.square, -0.5, 0.5, False)

    def test_square_contains_given_point_to_above_returns_false(self):
        self.check_contains(self.square, 0.5, 1.5, False)

    def test_square_contains_given_point_to_below_returns_false(self):
        self.check_contains(self.square, 0.5, -0.5, False)

    def test_contains_given_diamond_and_center_returns_true(self):
        self.check_contains(self.diamond, 0.5, 0.5, True)

    def test_diamond_contains_given_point_to_left_returns_false(self):
        self.check_contains(self.diamond, 1.5, 0.5, False)

    def test_diamond_contains_given_point_to_right_returns_false(self):
        self.check_contains(self.diamond, -0.5, 0.5, False)

    def test_diamond_contains_given_point_to_above_returns_false(self):
        self.check_contains(self.diamond, 0.5, 1.5, False)

    def test_diamond_contains_given_point_to_below_returns_false(self):
        self.check_contains(self.diamond, 0.5, -0.5, False)

    def test_diamond_contains_given_point_inside_box_outside_diamond(self):
        self.check_contains(self.diamond, 0.76, 0.25, False)

    def test_diamond_contains_given_point_on_boundary_returns_true(self):
        self.check_contains(self.diamond, 0.75, 0.25, True)

    def check_contains(self, grid_cell, x, y, expect):
        result = grid_cell.contains(x, y)
        self.assertEqual(expect, result)


class TestCellFromPositions(unittest.TestCase):
    def test_from_positions_given_regular_cell(self):
        lons = np.array([[1, 1, 1],
                         [2, 2, 2],
                         [3, 3, 3]])
        lats = np.array([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3]])
        positions = np.dstack((lons, lats)).astype("d")
        i, j = 1, 1
        fixture = cell.Cell.from_positions(positions, i, j)
        result = fixture.polygon
        expect = [(2, 2), (3, 2), (3, 3), (2, 3)]
        np.testing.assert_array_almost_equal(expect, result)

    def test_from_positions_given_different_cell(self):
        longitudes, latitudes = np.meshgrid([100, 104, 106, 107],
                                            [80, 83, 84],
                                            indexing="ij")
        positions = np.dstack((longitudes,
                               latitudes)).astype("d")
        i, j = 1, 1
        fixture = cell.Cell.from_positions(positions, i, j)
        result = fixture.polygon
        expect = [(104, 83),
                  (106, 83),
                  (106, 84),
                  (104, 84)]
        np.testing.assert_array_almost_equal(expect, result)

    def test_from_positions_given_cell_containing_dateline(self):
        lons = np.array([[178.5, 178.5, 178.5],
                         [179.5, 179.5, 179.5],
                         [-179.5, -179.5, -179.5]])
        lats = np.array([[1, 2, 3],
                         [1, 2, 3],
                         [1, 2, 3]])
        positions = np.dstack((lons, lats))
        i, j = 1, 1
        fixture = cell.Cell.from_positions(positions, i, j)
        self.assertIsInstance(fixture, cell.Dateline)


class TestCellCenter(unittest.TestCase):
    def setUp(self):
        self.unit_square = np.array([(0, 0),
                                     (1, 0),
                                     (1, 1),
                                     (0, 1)])
        self.centered_unit_square = np.array([(-0.5, -0.5),
                                              (+0.5, -0.5),
                                              (+0.5, +0.5),
                                              (-0.5, +0.5)])
        self.translated_unit_square = self.unit_square + [1, 0]

    def test_center_given_unit_square_returns_adjusted_center(self):
        self.check_center(self.unit_square,
                          (0.5, 0.500057))

    def test_center_given_translated_unit_square_returns_adjusted_center(self):
        self.check_center(self.translated_unit_square,
                          (1.5, 0.500057))

    def test_center_given_centered_unit_square(self):
        self.check_center(self.centered_unit_square, (0., 0.))

    def test_center_given_great_circle_center_at_north_pole(self):
        points = np.array([(0, 0),
                           (90, 45),
                           (0, 90),
                           (-90, 45)])
        self.check_center(points, (0, 90))

    def test_center_given_general_case(self):
        points = [(100, 80),
                  (104, 80),
                  (104, 83),
                  (100, 83)]
        expect = (102, 81.767828)  # Sphericity of Earth
        self.check_center(points, expect)

    def check_center(self, polygon, expect):
        fixture = cell.Cell(np.asarray(polygon, dtype="d"))
        result = fixture.center
        np.testing.assert_array_almost_equal(expect, result)


class TestDateline(unittest.TestCase):
    def setUp(self):
        vertices = np.array([(179, 0),
                             (-179, 0),
                             (-179, 1),
                             (179, 1)],
                            dtype="d")
        self.fixture = cell.Dateline(vertices)

    def test_contains_point_east_of_dateline_inside_cell_returns_true(self):
        self.check_contains(self.fixture, -179.5, 0.5, True)

    def test_contains_point_west_of_dateline_inside_cell_returns_true(self):
        self.check_contains(self.fixture, 179.5, 0.5, True)

    def test_contains_point_west_of_dateline_outside_cell_returns_false(self):
        self.check_contains(self.fixture, 178.5, 0.5, False)

    def test_contains_point_north_of_bounding_box_returns_false(self):
        self.check_contains(self.fixture, 179.5, 1.5, False)

    def check_contains(self, grid_cell, longitude, latitude, expect):
        result = grid_cell.contains(longitude, latitude)
        self.assertEqual(expect, result)


class TestSplitCell(unittest.TestCase):
    def test_split_cell_given_dateline_cutting_two_points(self):
        points = [(+179, 0),
                  (-179, 0),
                  (-179, 1),
                  (+179, 1)]
        west = [(+179, 0),
                (+180, 0),
                (+180, 1),
                (+179, 1)]
        east = [(-180, 0),
                (-179, 0),
                (-179, 1),
                (-180, 1)]
        result = cell.split_cell(points)
        expect = west, east
        self.assertSplitEqual(expect, result)

    def test_split_cell_given_dateline_cutting_one_point(self):
        points = [(+179, 0),
                  (-179, 1),
                  (+179, 2),
                  (+178, 1)]
        west = [(+179, 0),
                (+180, 0.5),
                (+180, 1.5),
                (+179, 2),
                (+178, 1)]
        east = [(-180, 0.5),
                (-179, 1),
                (-180, 1.5)]
        result = cell.split_cell(points)
        expect = west, east
        self.assertSplitEqual(expect, result)

    def assertSplitEqual(self, expect, result):
        west_result, east_result = result
        west_expect, east_expect = expect
        np.testing.assert_array_equal(west_expect, west_result)
        np.testing.assert_array_equal(east_expect, east_result)


class TestSameCell(unittest.TestCase):
    def test_same_cell_given_cells_with_same_vertices_returns_true(self):
        cell_1 = cell.Cell(np.asarray([(0, 0),
                                       (0, 1),
                                       (1, 1),
                                       (1, 0)], dtype="d"))
        cell_2 = cell.Cell(np.asarray([(0, 0),
                                       (0, 1),
                                       (1, 1),
                                       (1, 0)], dtype="d"))
        result = cell.same(cell_1, cell_2)
        expect = True
        self.assertEqual(expect, result)

    def test_same_cell_given_cells_with_different_vertices_returns_false(self):
        cell_1 = cell.Cell(np.asarray([(0, 0),
                                       (0, 1),
                                       (1, 1),
                                       (1, 0)], dtype="d"))
        cell_2 = cell.Cell(np.asarray([(0, 0),
                                       (0, 1),
                                       (1, 1),
                                       (1, 0.1)], dtype="d"))
        result = cell.same(cell_1, cell_2)
        expect = False
        self.assertEqual(expect, result)


class TestCellRepr(unittest.TestCase):
    def test_repr(self):
        data = np.zeros((2, 2), dtype="d")
        result = cell.Cell(data).__repr__()
        expect = "Cell({})".format(repr(data))
        self.assertEqual(expect, result)
