"""

Observation operator
====================

A simple Python based framework for assessing models. The library distributes
a :class:`ObservationOperator` class which
performs the 2D/3D interpolation from model space to observation space.

.. autoclass:: ObservationOperator
   :members:

.. automodule:: obsoper.interpolate
   :members:

.. automodule:: obsoper.horizontal
   :members:

.. automodule:: obsoper.vertical
   :members:

.. automodule:: obsoper.window
   :members:

.. automodule:: obsoper.grid
   :members:

.. automodule:: obsoper.exceptions
   :members:

"""
from .version import __version__
from .core import ObservationOperator
from .window import TimeWindow
from .interpolate import Tripolar
from .vertical import Section
from .orca import (remove_halo,
                   north_fold)
from .coordinates import cartesian
from .domain import Domain, point_in_polygon
