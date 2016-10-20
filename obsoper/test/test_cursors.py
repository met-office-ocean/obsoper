# pylint: disable=missing-docstring, invalid-name
import unittest
from obsoper import (cursors,
                     orca)


class TestNorthFold(unittest.TestCase):
    def test_getitem_given_point_returns_point_on_opposite_side(self):
        longitudes = [0, 0]
        latitudes = [0, 0]
        fixture = orca.north_fold(longitudes,
                                  latitudes)
        result = fixture[0]
        expect = 1
        self.assertEqual(expect, result)

    def test_getitem_given_multiple_pairs_of_points(self):
        longitudes = [0, 1, 2, 2, 1, 0]
        latitudes = [1, 2, 0, 0, 2, 1]
        fixture = orca.north_fold(longitudes,
                                  latitudes)
        result = fixture[1]
        expect = 4
        self.assertEqual(expect, result)


class TestTripolar(unittest.TestCase):
    def setUp(self):
        self.ni, self.nj = 4, 10
        fold = orca.north_fold(longitudes=[10, 20, 10, 20],
                               latitudes=[85, 85, 85, 85])
        self.fixture = cursors.Tripolar(self.ni, self.nj, fold)

    def test_move_given_null_move(self):
        self.check_move(i=0, j=0, di=0, dj=0, expect=(0, 0))

    def test_move_given_dj_increments_j_coordinate(self):
        self.check_move(i=0, j=0, di=0, dj=1, expect=(0, 1))

    def test_move_given_di_increments_i_coordinate(self):
        self.check_move(i=0, j=0, di=1, dj=0, expect=(1, 0))

    def test_move_given_j_and_dj_returns_sum_j_dj(self):
        self.check_move(i=0, j=1, di=0, dj=1, expect=(0, 2))

    def test_move_given_i_and_di_returns_sum_i_di(self):
        self.check_move(i=1, j=0, di=1, dj=0, expect=(2, 0))

    def test_move_given_i_on_eastern_edge_wraps_around(self):
        self.check_move(i=self.ni - 1, j=0, di=1, dj=0, expect=(0, 0))

    def test_move_given_i_on_western_edge_wraps_around(self):
        self.check_move(i=0, j=0, di=-1, dj=0, expect=(self.ni - 1, 0))

    def test_move_across_north_fold_jumps_in_i(self):
        self.check_move(i=0, j=self.nj - 2, di=0, dj=1,
                        expect=(2, self.nj - 2))

    def check_move(self, i, j, di, dj, expect):
        result = self.fixture.move(i, j, di, dj)
        self.assertEqual(expect, result)
