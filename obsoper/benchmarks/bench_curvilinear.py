"""Benchmark curvilinear interpolation algorithm"""
# pylint: disable=missing-docstring, invalid-name
import netCDF4
import bench
import util
from obsoper.interpolate import Curvilinear


class BenchmarkCurvilinear(bench.Suite):
    def setUp(self):
        for path in ["sample_class4.nc",
                     "sample_prodm.nc"]:
            util.grab(path)

        with netCDF4.Dataset("data/sample_class4.nc") as dataset:
            self.observed_longitudes = dataset.variables["longitude"][:]
            self.observed_latitudes = dataset.variables["latitude"][:]

        with netCDF4.Dataset("data/sample_prodm.nc") as dataset:
            self.grid_longitudes = dataset.variables["TLON"][:].T
            self.grid_latitudes = dataset.variables["TLAT"][:].T
            self.grid_values = dataset.variables["aice"][0, :, :].T

        # Correct longitudes [0, 360) to [-180, 180)
        self.grid_longitudes[self.grid_longitudes > 180] -= 360.

    def bench_curvilinear_given_10e0_observations(self):
        self.run_curvilinear(1)

    def bench_curvilinear_given_10e1_observations(self):
        self.run_curvilinear(10)

    def bench_curvilinear_given_10e2_observations(self):
        self.run_curvilinear(100)

    def bench_curvilinear_given_10e3_observations(self):
        self.run_curvilinear(1000)

    def bench_curvilinear_given_10e4_observations(self):
        self.run_curvilinear(10**4)

    def bench_curvilinear_given_10e5_observations(self):
        self.run_curvilinear(10**5)

    def bench_curvilinear_given_10e6_observations(self):
        self.run_curvilinear(10**6)

    def run_curvilinear(self, number):
        # Simulate initialising interpolator
        interpolator = Curvilinear(self.grid_longitudes,
                                   self.grid_latitudes,
                                   self.observed_longitudes[:number],
                                   self.observed_latitudes[:number])
        for _ in range(3):
            # Simulate interpolating multiple forecasts
            interpolator.interpolate(self.grid_values)
