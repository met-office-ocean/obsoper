"""
Geographic coordinate conversion methods
"""
import numpy as np


def cartesian(longitudes, latitudes):
    """Convert to cartesian coordinate system

    .. note:: Assumes radius of sphere is 1

    :param longitudes: array of longitudes in degrees
    :param latitudes: array of latitudes in degrees
    :returns: (x, y, z) tuple of Cartesian coordinates arrays
    """
    # pylint: disable=invalid-name
    longitudes = np.deg2rad(longitudes)
    latitudes = np.deg2rad(latitudes)

    x = np.cos(longitudes) * np.cos(latitudes)
    y = np.sin(longitudes) * np.cos(latitudes)
    z = np.sin(latitudes)
    return x, y, z
