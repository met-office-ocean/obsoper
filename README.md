# Python observation operator

Map ocean forecasts from model space to observation space quickly and efficiently.

[![Build Status](https://travis-ci.org/met-office-ocean/obsoper.svg?branch=master)](https://travis-ci.org/met-office-ocean/obsoper)

## Installation

There are two ways to install the package, via the Python Package Index (PYPI) and manually from a repository clone.

### PYPI

The [obsoper](https://pypi.python.org/pypi/obsoper) package is available on [PYPI](https://pypi.python.org). To install
simply call `pip install --user`. It is recommended to use the `--user` flag to prevent contamination of the system Python build.

### Clone repository

Alternatively, it is possible to build from source by cloning this repository. The package itself uses setuptools to install and
can be installed using `python setup.py install --user`.

If installation was successful it should be possible to import the package without error.

```python
>>> import obsoper
```

## Basic usage

In what follows, variables prepended with `grid_` or `observed_` are specified as N-dimenional numpy arrays.

### Regular longitude/latitude grids

Evenly spaced longitude/latitude grids can be specified by 1 or 2 dimensional arrays. If the full 2 dimensional
grid is specified then only `grid_longitudes[:, 0]` and `grid_latitudes[0, :]` are used to define the search criteria.

```python
>>> operator = obsoper.Operator(grid_longitudes,
...                             grid_latitudes,
...                             observed_longitudes,
...                             observed_latitudes)
```

Once the default observation operator has seen the grid definition it knows
the grid extent and how to select indices surrounding points in space.

```python
>>> result = operator.interpolate(grid_sst)
```

### Tri-polar grids

Tri-polar ORCA grids are more complicated than regular grids in a number of ways. As well as having irregularly shaped cells there is also a fold joining the two northern poles. Efficiently searching and interpolating on these grids can be problematic.

Typical usage involves a fixed set of observations with multiple diagnostic fields being compared against iteratively. To speed computation giving the tri-polar operator as much information as possible up front reduces repetitive computation later in the process.

```python
>>> operator = obsoper.Operator(grid_longitudes.T,
...                             grid_latitudes.T,
...                             observed_longitudes,
...                             observed_latitudes,
...                             layout="tripolar",
...                             has_halo=True)
```

**Note:** Grid longitude and latitude arrays must be shaped (x, y), where **x** represents longitude and **y** represent latitude directions. NEMO diagnostics are typically stored (t, z, y, x) appropriate transpose operations should be made prior to interpolation.

Once the operator has been trained on a set of data, it is then possible to iteratively interpolate a collection of forecasts.

```python
>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast)
```

Interpolated model counterparts can then be written to a file or analysed further to generate plots.

### Regional models

Regional models with non-trivial boundaries and rotated coordinate systems can be interpolated by specifying `layout="regional"`.

```python
>>> operator = obsoper.Operator(grid_longitudes.T,
...                             grid_latitudes.T,
...                             observed_longitudes,
...                             observed_latitudes,
...                             layout="regional")
```

### Interpolating profiles

Vertical interpolation is triggered by specifying `grid_depths` and `observed_depths` keyword arguments. Using our
regional model example above it is easy to extend the call to the `obsoper.Operator` constructor to allow
model and profile depths.

```python
>>> operator = obsoper.Operator(grid_longitudes.T,
...                             grid_latitudes.T,
...                             obs_longitudes,
...                             obs_latitudes,
...                             grid_depths=grid_depths.T,
...                             observed_depths=obs_depths,
...                             layout="regional")
```

#### Vertical coordinates

Both Z-level and S-level models are handled seamlessly behind the scenes. The operator knows if a grid has the same vertical levels at each grid point or horizontally varying levels based on the dimensionality of `grid_depths` keyword.

## Full documentation

More comprehensive documentation can be found at [obsoper.readthedocs.io](http://obsoper.readthedocs.io).

