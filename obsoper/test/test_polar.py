# pylint: disable=missing-docstring, invalid-name
import unittest
from obsoper import polar


class TestPolarSearch(unittest.TestCase):
    def test_polar_search(self):
        polar.search()
