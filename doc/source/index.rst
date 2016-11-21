.. Observation operator documentation master file, created by
   sphinx-quickstart on Mon Dec 14 16:32:50 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Observation operator
====================

Light weight fast interpolator that can map ocean model forecasts to
observation locations.

Basic usage
-----------

The obsoper package contains tools to simplify interpolation to observed locations.

**Old API**

For example, to interpolate a sea level anomaly analysis to altimeter tracks, the following steps may be taken.

>>> operator = obsoper.ObservationOperator(grid_lons, grid_lats)
>>> result = operator.interpolate(sla_analysis, track_lons, track_lats)

To process multiple forecasts, the operator instance can be reused.

>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast, track_lons, track_lats)

**New API**

For example, to interpolate a sea level anomaly analysis to altimeter tracks, the following steps may be taken.

>>> operator = obsoper.Operator(grid_lons, grid_lats, track_lons, track_lats)
>>> result = operator.interpolate(sla_analysis)

To process multiple forecasts, the operator instance can be reused.

>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast)

In the above examples, only surface fields have been used. To interpolate
to observed profiles, add grid and observed depths arrays. Z-level models
have 1D depth arrays, whereas S-level models have 3D depths.

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
...                             has_halo=True,
...                             search="tripolar",
...                             boundary="band")
>>> result = operator.interpolate(temperature_analysis)


Contents:

.. toctree::
   :maxdepth: 2

   algorithm
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

