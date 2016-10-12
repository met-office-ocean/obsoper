# pylint: disable=missing-docstring, invalid-name
import unittest
import run


class TestParseArgs(unittest.TestCase):
    def test_modules_given_names(self):
        self.check_parse_args(["A", "B"], "modules", ["A", "B"])

    def test_method_default_none(self):
        self.check_parse_args([], "method", None)

    def test_method_given_string(self):
        self.check_parse_args(["--method", "name"], "method", "name")

    def test_save_returns_true(self):
        self.check_parse_args(["--save"], "save", True)

    def check_parse_args(self, argv, attribute, expect):
        result = getattr(run.parse_args(argv), attribute)
        self.assertEqual(expect, result)
