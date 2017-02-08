# pylint: disable=missing-docstring, invalid-name
import unittest
import obsoper


class TestVersion(unittest.TestCase):
    def test_version(self):
        self.assertEqual(obsoper.__version__, "0.0.8")
