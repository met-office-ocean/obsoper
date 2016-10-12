Python observation operator
===========================

Maps ocean forecasts from model space to observation space

Installation
------------

Python package distributed with setuptools. Requires Cython to build
extension modules.

Can be installed directly from setup.py

```bash
:> python setup.py install
```

Or with pip

```bash
:> python setup.py bdist_wheel
:> pip install dist/obsoper-0.0.2-cp27-none-linux_x86_64.whl
```

If installation was successful it should be possible to import the package without error.

```python
>>> import obsoper
```

Basic usage
-----------

Regular latitude/longitude grids can be specified by 1 dimensional arrays.

```python
>>> nlon, nlat = 13, 10
>>> grid_longitudes = np.linspace(-180, 180, nlon)
>>> grid_latitudes = np.linspace(-90, 90, nlat)
>>> operator = obsoper.ObservationOperator(grid_longitudes, grid_latitudes)
```

Once the default observation operator has seen the 1 dimensional grid definition it knows
the grid extent and how to select indices surrounding a point in space.

```python
>>> grid_sst = np.full((nlon, nlat), 30) + np.random.randn(130).reshape((nlon, nlat))
>>> observed_lons = np.array([100])
>>> observed_lats = np.array([10])
>>> operator.interpolate(grid_sst, observed_lons, observed_lats)
masked_array(data = [28.76232843679889],
             mask = [False],
       fill_value = 1e+20)
```

Tri-polar ORCA grids are more complicated than regular grids in a number of ways. As well as having irregularly shaped cells there is also a fold joining the two northern poles. Efficiently searching and interpolating on these grids can be problematic.

Typical usage involves a fixed set of observations with multiple diagnostic fields being compared against iteratively. To speed computation giving the tri-polar operator as much information as possible up front reduces repetitive computation later in the process.

```python
>>> operator = obsoper.Tripolar(grid_longitudes.T, grid_latitudes.T, obs_longitudes, obs_latitudes)
```

**Note:** Grid longitude and latitude arrays must be shaped (x, y), where x represents longitude and y represent latitude directions. NEMO diagnostics are typically stored (t, z, y, x) appropriate transpose operations should be made prior to interpolation.

Once the operator has been trained on a set of data, it is then possible to iteratively interpolate a collection of forecasts.

```python
>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast)
```

Interpolated model counterparts can then be written to a file or analysed further to generate plots.

