"""
Model domains
=============

Regular latitude/longitude
--------------------------

For regular latitude/longitude grids a simple bounding box test is all
that is required to determine if a point lies inside the domain.

Irregular boundaries
--------------------

Regional ocean models with irregular boundaries can perform an additional
point in polygon check to determine if a point is inside the domain.

Global models
-------------

Ocean models of global extent typically have a southern boundary, since
Antarctica is a land mass covering the South Pole. A North/South extent
check may be sufficient to determine whether a point belongs to the domain or
not.
"""
import numpy as np
from . import box


class Domain(object):
    """Grid domain definition"""
    def __init__(self, longitudes, latitudes):
        self.bounding_box = box.Box(np.min(longitudes),
                                    np.max(longitudes),
                                    np.min(latitudes),
                                    np.max(latitudes))

    def contains(self, longitudes, latitudes):
        """check observations are contained within domain"""
        return self.bounding_box.inside(longitudes, latitudes)
