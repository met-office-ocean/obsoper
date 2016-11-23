"""
Custom Exceptions
=================

Collection of custom exceptions used throughout the package.
"""


class SearchFailed(Exception):
    """Indicate grid search failure"""


class StepNotFound(Exception):
    """Indicate great circle arc overlapping failure"""


class NotInGrid(Exception):
    """indicates point lies outside of grid"""


class TooFewKnots(Exception):
    """indicates interpolator can not create spline"""


class RequiredArgument(Exception):
    """Indicate required argument is missing from method call"""


class UnknownLayout(Exception):
    """Indicate ocean model layout isn't recognised or supported"""
