# Python observation operator

Maps ocean forecasts from model space to observation space

[![Build Status](https://travis-ci.org/met-office-ocean/obsoper.svg?branch=master)](https://travis-ci.org/met-office-ocean/obsoper)

## Installation

Python package distributed with setuptools. Requires Cython to build
extension modules.

Can be installed directly from setup.py

```bash
:> python setup.py install --user
```

Or with pip

```bash
:> python setup.py bdist_wheel
:> pip install dist/obsoper-${VERSION}-cp27-none-linux_x86_64.whl
```

If installation was successful it should be possible to import the package without error.

```python
>>> import obsoper
```

## Basic usage

### Regular lat/lon grids

Regular latitude/longitude grids can be specified by 1 dimensional arrays.

```python
>>> operator = obsoper.Operator(gridded_longitudes,
...                             gridded_latitudes,
...                             observed_longitudes,
...                             observed_latitudes)
```

Once the default observation operator has seen the 1 dimensional grid definition it knows
the grid extent and how to select indices surrounding a point in space.

```python
>>> result = operator.interpolate(gridded_sst)
```

### Tri-polar grids

Tri-polar ORCA grids are more complicated than regular grids in a number of ways. As well as having irregularly shaped cells there is also a fold joining the two northern poles. Efficiently searching and interpolating on these grids can be problematic.

Typical usage involves a fixed set of observations with multiple diagnostic fields being compared against iteratively. To speed computation giving the tri-polar operator as much information as possible up front reduces repetitive computation later in the process.

```python
>>> operator = obsoper.Operator(grid_longitudes.T,
...                             grid_latitudes.T,
...                             obs_longitudes,
...                             obs_latitudes,
...                             layout="tripolar",
...                             has_halo=True)
```

**Note:** Grid longitude and latitude arrays must be shaped (x, y), where x represents longitude and y represent latitude directions. NEMO diagnostics are typically stored (t, z, y, x) appropriate transpose operations should be made prior to interpolation.

Once the operator has been trained on a set of data, it is then possible to iteratively interpolate a collection of forecasts.

```python
>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast)
```

Interpolated model counterparts can then be written to a file or analysed further to generate plots.

## Non-regular regional models

Regional models with non-trivial boundaries and rotated coordinate systems can be interpolated using the following keyword arguments.

```python
>>> operator = obsoper.Operator(grid_longitudes.T,
...                             grid_latitudes.T,
...                             obs_longitudes,
...                             obs_latitudes,
...                             layout="regional")
```

# Cubic spline vertical interpolation

Vertical interpolation is triggered by specifying grid_depths and observed_depths keyword arguments. 

```python
>>> operator = obsoper.Operator(grid_longitudes.T,
...                             grid_latitudes.T,
...                             obs_longitudes,
...                             obs_latitudes,
...                             grid_depths=grid_depths.T,
...                             obs_depths=obs_depths,
...                             layout="regional")
```

## Full documentation

More comprehensive documentation can be found at [obsoper.readthedocs.io](http://obsoper.readthedocs.io).

