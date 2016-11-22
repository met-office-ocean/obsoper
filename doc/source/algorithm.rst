
Algorithms
==========

Before taking a detailed look at the concrete classes and methods included
in the package it is worthwhile to take a brief overview of the algorithms
included in the package.

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

**Tri-polar**

For tri-polar grids a KD Tree algorithm is used to locate the "nearest" grid
point in longitude/latitude space. Then to reduce the ammount of checking needed
to locate the four grid points surrounding the observation an algorithm
was designed to "walk" the grid in the direction of the observed point.

Interpolation
-------------

**Horizontal**

Bilinear interpolation is simple on a unit square but slightly more involved
on more complicated geometries.

* Unit square
* Arbitrary quadrilateral

**Vertical**

The vertical interpolation method used throughout the package is cubic spline
interpolation.


