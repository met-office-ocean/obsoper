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

For example, to interpolate a sea level anomaly analysis to Jason-2 altimeter tracks, the following steps may be taken.

>>> operator = obsoper.ObservationOperator(grid_lons, grid_lats)
>>> result = operator.interpolate(sla_analysis, track_lons, track_lats)

To process multiple forecasts, the operator instance can be reused.

>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast, track_lons, track_lats)

**New API**

For example, to interpolate a sea level anomaly analysis to Jason-2 altimeter tracks, the following steps may be taken.

>>> operator = obsoper.Operator.from_arrays(grid_lons, grid_lats, track_lons, track_lats)
>>> result = operator.interpolate(sla_analysis)

To process multiple forecasts, the operator instance can be reused.

>>> for forecast in forecasts:
...     counterparts = operator.interpolate(forecast)

In the above examples, only numpy arrays and a single class were used to interpolate to observed locations. A
more complicated use case would include depths.

>>> operator = obsoper.Operator.from_arrays(grid_lons, grid_lats, argo_lons, argo_lats, grid_depths=grid_depths, observed_depths=argo_depths)
>>> result = operator.interpolate(temperature_analysis)

Clearly, this would become unmanageable if we add extra parameters, such as time. An alternative usage would be to use abstractions
for the grid and observations.

>>> grid = obsoper.Grid(grid_lons, grid_lats, grid_depths, layout="tripolar")
>>> positions = obsoper.Positions(argo_lons, argo_lats, argo_depths)
>>> operator = obsoper.Operator(grid, positions)

Contents:

.. toctree::
   :maxdepth: 2

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

