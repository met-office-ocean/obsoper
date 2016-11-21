
Algorithms
==========

There are several steps to take when mapping forecasts to observed locations. Each
step has it's own family of algorithms.

Domain membership
-----------------

Check whether a position is inside a model domain.

* Latitude band - select observation between parallels
* Bounding box - select observations between parallels and meridians
* Point in polygon - use domain edge as a polygon

A latitude band filter can be constructed via
:class:`obsoper.domain.LatitudeBand`. This simple class detects observations
that are north of some latitude minimum and north of some latitude maximum.

The next step up in complexity from a latitude band is a longitude/latitude
box, available via :class:`obsoper.domain.Box`. In addition to north/south
screening and east/west check is also performed.

Finally, the most thorough but also complex algorithm, relies on some topology
of two dimensional shapes. For any non self-intersecting polygon, a line
passing through a point of interest intersects the edges of the polygon
multiple times. If there are an odd number of intersections on either side
of the point in question then the point is considered to be inside the domain.
See, :class:`obsoper.domain.Polygon`.

Grid searching
--------------

Navigate or otherwise interrogate the grid layout to detect 4 model grid points
surrounding a position of interest.

* Simple index dereferencing of a regular lon/lat grid
* KD tree in longitude/latitude space
* KD tree in Cartesian space

Interpolation
-------------

Bilinear interpolation is simple on a unit square but slightly more involved
on more complicated geometries.

* Unit square
* Arbitrary quadrilateral


