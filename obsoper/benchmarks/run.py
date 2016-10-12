#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Benchmark runner"""
import importlib
import glob
import argparse
import bench


def parse_args(argv=None):
    """command line parser"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*",
                        help="Benchmark module names")
    parser.add_argument("--save", action="store_true",
                        help="Keep a run as a reference.")
    parser.add_argument("--method",
                        help="Select particular benchmark method.")
    return parser.parse_args(args=argv)


def main():
    """Run benchmark suites"""
    args = parse_args()
    if len(args.modules) == 0:
        args.modules = sorted(glob.glob("bench_*.py"))

    for module in args.modules:
        importlib.import_module(module.replace(".py", ""))

    bench.run(args.save, args.method)


if __name__ == '__main__':
    main()
