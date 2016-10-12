"""Benchmark Search algorithm"""
# pylint: disable=missing-docstring, invalid-name
import netCDF4
import bench
import util
import obsoper.grid


class BenchmarkRealData(bench.Suite):
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
            self.grid_values = dataset.variables["aice"][:]

        # Correct longitudes [0, 360) to [-180, 180)
        self.grid_longitudes[self.grid_longitudes > 180] -= 360.

    def bench_interpolate_given_10e0_observations(self):
        self.run_interpolate(1)

    def bench_interpolate_given_10e1_observations(self):
        self.run_interpolate(10)

    def bench_interpolate_given_10e2_observations(self):
        self.run_interpolate(100)

    def bench_interpolate_given_10e3_observations(self):
        self.run_interpolate(1000)

    def bench_interpolate_given_10e4_observations(self):
        self.run_interpolate(10**4)

    def bench_interpolate_given_10e5_observations(self):
        self.run_interpolate(10**5)

    def run_interpolate(self, number):
        interpolator = obsoper.grid.Search(self.grid_longitudes,
                                           self.grid_latitudes)
        interpolator.lower_left(self.observed_longitudes[:number],
                                self.observed_latitudes[:number])
