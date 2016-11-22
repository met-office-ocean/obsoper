
Basic usage
===========

The obsoper package contains tools to simplify interpolation to observed locations.

Surface
-------

To interpolate a surface parameter like a sea level anomaly analysis to altimeter tracks the following steps may be taken.

>>> operator = obsoper.Operator(grid_lons, grid_lats, track_lons, track_lats)
>>> result = operator.interpolate(sla_analysis)

To process multiple forecasts, the operator instance can be reused.

>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast)


Full depth
----------

To interpolate vertically as well as horizontally additional information
needs to be given to the :class:`obsoper.Operator`. To interpolate a 3D temperature field to Argo profiles simply add grid
and observed depths arrays.

>>> operator = obsoper.Operator(grid_lons,
...                             grid_lats,
...                             argo_lons,
...                             argo_lats,
...                             grid_depths=grid_depths,
...                             observed_depths=argo_depths)
>>> result = operator.interpolate(temperature_analysis)

To interpolate from a tripolar grid, care must be taken to first identify
if the grid has a halo.

>>> operator = obsoper.Operator(grid_lons,
...                             grid_lats,
...                             argo_lons,
...                             argo_lats,
...                             grid_depths=grid_depths,
...                             observed_depths=argo_depths,
...                             layout="tripolar",
...                             has_halo=True)
>>> result = operator.interpolate(temperature_analysis)


Regional models
---------------

Complex domain regional ocean models are handled by the keyword
``layout='regional'``. For example, to interpolate a shelf seas
model with irregular boundaries and rotated longitudes and latitudes.

>>> operator = obsoper.Operator(grid_lons,
...                             grid_lats,
...                             argo_lons,
...                             argo_lats,
...                             grid_depths=grid_depths,
...                             observed_depths=argo_depths,
...                             layout="regional")
>>> result = operator.interpolate(shelf_seas_analysis)

Multiple forecasts can be interpolated rapidly as before.

>>> for forecast in shelf_seas_forecasts:
...     counterparts = operator.interpolate(forecast)

.. note:: To interpolate a different collection of observations a new operator
          instance must be constructed.


