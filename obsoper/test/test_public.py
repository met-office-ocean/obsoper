# pylint: disable=missing-docstring, invalid-name
import os
import unittest
import pkg_resources

HAS_NETCDF4 = True
try:
    import netCDF4
except ImportError:
    HAS_NETCDF4 = False

import numpy as np
import obsoper

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
ORCA12_FILE = os.path.join(SCRIPT_DIR, "data/ostdemo_orca12.nc")
ORCA025_FILE = os.path.join(SCRIPT_DIR, "data/orca025_grid.nc")
ORCA025_CICE_FILE = os.path.join(SCRIPT_DIR, "data/orca025_cice.nc")


def locate_file(name):
    """Locate file inside test suite"""
    return pkg_resources.resource_filename("obsoper.test",
                                           os.path.join("data", name))

if HAS_NETCDF4:
    @unittest.skip("slow test")
    class TestTripolar(unittest.TestCase):
        def setUp(self):
            orca025_grid = locate_file("orca025_grid.nc")
            self.grid_longitudes = self.read_variable(orca025_grid, "nav_lon")
            self.grid_latitudes = self.read_variable(orca025_grid, "nav_lat")

            self.observed_longitudes = np.array([100])
            self.observed_latitudes = np.array([10])

            shape = self.grid_longitudes.T.shape
            self.constant = 30.
            self.constant_field = np.full(shape, self.constant)

        def read_variable(self, path, name):
            with netCDF4.Dataset(path) as dataset:
                return np.ma.asarray(dataset.variables[name][:])

        def test_tripolar_interpolation_given_constant_surface_field(self):
            fixture = obsoper.Tripolar(self.grid_longitudes.T,
                                       self.grid_latitudes.T,
                                       self.observed_longitudes,
                                       self.observed_latitudes)
            result = fixture.interpolate(self.constant_field)
            expect = np.array([self.constant])
            np.testing.assert_array_equal(expect, result)


class TestORCA025(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA025_FILE) as dataset:
            cls.grid_longitudes = dataset.variables["nav_lon"][:]
            cls.grid_latitudes = dataset.variables["nav_lat"][:]

    def setUp(self):
        shape = self.grid_longitudes.T.shape
        self.constant = 30.
        self.constant_field = np.full(shape, self.constant)

    def test_tripolar_interpolation_given_constant_surface_field(self):
        lons, lats = [100], [10]
        fixture = obsoper.Tripolar(self.grid_longitudes.T,
                                   self.grid_latitudes.T,
                                   lons,
                                   lats)
        result = fixture.interpolate(self.constant_field)
        expect = np.array([self.constant])
        np.testing.assert_array_equal(expect, result)

    def test_radial_algorithm_exhaustion(self):
        lons, lats = [-9.9305896759], [-44.4005584717]
        fixture = obsoper.Tripolar(self.grid_longitudes.T,
                                   self.grid_latitudes.T,
                                   lons,
                                   lats)
        result = fixture.interpolate(self.constant_field)
        expect = np.array([self.constant])
        np.testing.assert_array_equal(expect, result)


@unittest.skip("implementing simpler algorithm")
class TestORCA025CICE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA025_CICE_FILE) as dataset:
            cls.grid_longitudes = dataset.variables["TLON"][:]
            cls.grid_latitudes = dataset.variables["TLAT"][:]

    def setUp(self):
        shape = self.grid_longitudes.T.shape
        self.constant = 30.
        self.constant_field = np.full(shape, self.constant)

    def test_tripolar_interpolation_given_constant_surface_field(self):
        lons, lats = [100], [10]
        fixture = obsoper.Tripolar(self.grid_longitudes.T,
                                   self.grid_latitudes.T,
                                   lons,
                                   lats)
        result = fixture.interpolate(self.constant_field)
        expect = np.array([self.constant])
        np.testing.assert_array_equal(expect, result)

    def test_radial_algorithm_exhaustion(self):
        lons, lats = [-9.9305896759], [-44.4005584717]
        fixture = obsoper.Tripolar(self.grid_longitudes.T,
                                   self.grid_latitudes.T,
                                   lons,
                                   lats)
        result = fixture.interpolate(self.constant_field)
        expect = np.array([self.constant])
        np.testing.assert_array_equal(expect, result)



class TestORCA12(unittest.TestCase):
    """Integration tests to confirm ORCA12 support

    .. note: ORCA12 nav_lat has missing data at index 1494 since
             it is the equator and the _FillValue is 0.
    """
    @classmethod
    def setUpClass(cls):
        with netCDF4.Dataset(ORCA12_FILE) as dataset:
            cls.orca12_lons = dataset.variables["nav_lon"][:]
            cls.orca12_lats = dataset.variables["nav_lat"][:]

    def setUp(self):
        self.grid_lons = np.ma.copy(self.orca12_lons)
        self.grid_lats = np.ma.copy(self.orca12_lats)
        self.field = np.ones(self.grid_lons.T.shape)

        # Fill in missing equator
        self.grid_lats[1494] = 0

    @unittest.skip("solving radial grid search problem")
    def test_interpolate_given_random_observation(self):
        """should work for all data between 90S, 90N and 180E, 180W"""
        sample_size = 10**4
        lons = np.random.uniform(-180, 180, sample_size)
        lats = np.random.uniform(-90, 90, sample_size)
        self.check_interpolate(lons, lats)

    def test_radial_algorithm_exhaustion(self):
        lons, lats = [150.012414668], [89.9564470382]
        self.check_interpolate(lons, lats)

    def check_interpolate(self, lons, lats):
        operator = obsoper.Operator(self.grid_lons.T,
                                    self.grid_lats.T,
                                    lons,
                                    lats,
                                    layout="tripolar",
                                    has_halo=True)
        result = operator.interpolate(self.grid_lons.T)
        expect = lons

        # Diagnose failure
        errors = np.ma.abs(result - expect)
        index = np.ma.argmax(errors)
        print result[index], lons[index], lats[index]

        np.testing.assert_array_almost_equal(expect, result)


if not HAS_NETCDF4 or os.path.exists(ORCA12_FILE):
    TestORCA12 = unittest.skip(TestORCA12)


class TestPublicInterface(unittest.TestCase):
    def test_remove_halo_is_accessible_from_library_import(self):
        self.assertEqual(obsoper.orca.remove_halo, obsoper.remove_halo)

    def test_north_fold_is_accessible_from_library_import(self):
        self.assertEqual(obsoper.orca.north_fold, obsoper.north_fold)

    def test_section_is_accessible_from_library_import(self):
        self.assertEqual(obsoper.vertical.Section, obsoper.Section)
