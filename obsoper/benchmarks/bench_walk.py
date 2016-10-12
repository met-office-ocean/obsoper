"""Benchmark Walk algorithm"""
import numpy as np
import bench
import obsoper.walk


class BenchmarkWalk(bench.Suite):
    def setUp(self):
        longitudes, latitudes = np.meshgrid([1, 2, 3],
                                            [1, 2, 3],
                                            indexing="ij")
        self.fixture = obsoper.walk.Walk.from_lonlats(longitudes,
                                                      latitudes)

    def bench_detect(self):
        for _ in range(10):
            self.fixture.detect((2.9, 2.9), i=0, j=0)
