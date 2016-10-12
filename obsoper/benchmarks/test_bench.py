# pylint: disable=missing-docstring, invalid-name
import unittest
import os
import json
import bench


class TestBenchmarkName(unittest.TestCase):
    def test_suite_name_given_class(self):
        result = bench.suite_name(TestBenchmarkName)
        expect = "{}.{}".format(TestBenchmarkName.__module__,
                                TestBenchmarkName.__name__)
        self.assertEqual(expect, result)


class TestLoad(unittest.TestCase):
    def setUp(self):
        self.json_path = "test.json"

    def tearDown(self):
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def test_load_given_no_data(self):
        result = bench.load(self.json_path)
        expect = {}
        self.assertEqual(expect, result)

    def test_load_given_single_entry(self):
        with open(self.json_path, "w") as handle:
            json.dump({"name": 1.0}, handle)

        result = bench.load(self.json_path)
        expect = {"name": 1.0}
        self.assertEqual(expect, result)


class TestSave(unittest.TestCase):
    def setUp(self):
        self.json_path = "test.json"

    def tearDown(self):
        if os.path.exists(self.json_path):
            os.remove(self.json_path)

    def test_save_adds_entry_to_json_file(self):
        bench.save("benchmark", 1.0, path=self.json_path)
        self.check_save({"benchmark": 1.0})

    def test_save_given_two_different_entries(self):
        bench.save("A", 1.0, self.json_path)
        bench.save("B", 2.0, self.json_path)
        self.check_save({"A": 1.0,
                         "B": 2.0})

    def test_save_given_two_entries_wth_same_key_returns_final_value(self):
        bench.save("A", 1.0, self.json_path)
        bench.save("A", 2.0, self.json_path)
        self.check_save({"A": 2.0})

    def check_save(self, expect):
        with open(self.json_path, "r") as handle:
            result = handle.read()
        expect = json.dumps(expect)
        self.assertEqual(expect, result)
