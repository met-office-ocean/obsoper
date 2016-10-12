# pylint: disable=missing-docstring, invalid-name
import unittest
import locate


class TestLocator(unittest.TestCase):
    def test_select(self):
        fixture = locate.Locator()
        result = list(fixture.select(["bench_method", "setUp"]))
        expect = ["bench_method"]
        self.assertEqual(expect, result)


class TestIsBenchmark(unittest.TestCase):
    def test_is_benchmark_given_non_benchmark_returns_false(self):
        self.check_is_benchmark("helper", False)

    def test_is_benchmark_given_benchmark_returns_true(self):
        self.check_is_benchmark("bench_method", True)

    def check_is_benchmark(self, method, expect):
        result = locate.Locator.is_benchmark(method)
        self.assertEqual(expect, result)
