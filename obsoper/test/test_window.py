# pylint: disable=missing-docstring, invalid-name
import unittest
import datetime as dt
import numpy as np
from obsoper.window import TimeWindow, EmptyWindow


class TestEmptyWindow(unittest.TestCase):
    def setUp(self):
        self.fixture = EmptyWindow()
        self.numbers = np.arange(3)

    def test_inside_always_returns_false(self):
        result = self.fixture.inside(self.numbers)
        expect = np.array([False, False, False], dtype=np.bool)
        np.testing.assert_array_equal(expect, result)

    def test_outside_always_returns_true(self):
        result = self.fixture.outside(self.numbers)
        expect = np.array([True, True, True], dtype=np.bool)
        np.testing.assert_array_equal(expect, result)


class TestTimeWindow(unittest.TestCase):
    def setUp(self):
        self.before = dt.datetime(2014, 12, 31, 0, 0, 0)
        self.start = dt.datetime(2015, 1, 1, 0, 0, 0)
        self.inside = dt.datetime(2015, 1, 1, 12, 0, 0)
        self.end = dt.datetime(2015, 1, 2, 0, 0, 0)
        self.after = dt.datetime(2015, 1, 3, 0, 0, 0)
        self.fixture = TimeWindow(self.start, self.end)

    def test_inside_given_time_inside_window_returns_true(self):
        self.check_inside(self.inside, True)

    def test_outside_given_time_inside_window_returns_false(self):
        self.check_outside(self.inside, False)

    def test_inside_given_time_after_end_returns_false(self):
        self.check_inside(self.after, False)

    def test_inside_given_time_before_start_returns_false(self):
        self.check_inside(self.before, False)

    def test_inside_given_start_returns_true(self):
        self.check_inside(self.start, True)

    def test_inside_given_end_returns_false(self):
        self.check_inside(self.end, False)

    def check_inside(self, time, expect):
        result = self.fixture.inside(np.asarray([time]))
        expect = np.array([expect])
        np.testing.assert_array_equal(expect, result)

    def check_outside(self, time, expect):
        result = self.fixture.outside(np.asarray([time]))
        expect = np.array([expect])
        np.testing.assert_array_equal(expect, result)

    def test_fromtext_given_start_returns_datetime(self):
        fixture = TimeWindow.fromtext("20150101", "20150102")
        result = fixture.start
        expect = self.start
        self.assertEqual(expect, result)

    def test_fromtext_given_end_returns_datetime(self):
        fixture = TimeWindow.fromtext("20150101", "20150102")
        result = fixture.end
        expect = self.end
        self.assertEqual(expect, result)

    def test_frominterval_given_start_returns_datetime(self):
        fixture = TimeWindow.frominterval(self.start, self.end - self.start)
        result = fixture.start
        expect = self.start
        self.assertEqual(expect, result)

    def test_frominterval_given_end_returns_datetime(self):
        fixture = TimeWindow.frominterval(self.start, self.end - self.start)
        result = fixture.end
        expect = self.end
        self.assertEqual(expect, result)
