"""Utility functions for benchmarking obsoper"""
import os
import subprocess


def grab(path):
    """Grab data file from MASS-R"""
    if not os.path.exists(os.path.join("data", path)):
        moo_dir = "moose:/adhoc/users/andrew.ryan/benchmark/obsoper/data/"
        url = os.path.join(moo_dir, path)
        subprocess.call(["moo", "get", url, "data/"])
