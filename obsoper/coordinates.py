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
    longitudes = radians(longitudes)
    latitudes = radians(latitudes)

    x = np.cos(longitudes) * np.cos(latitudes)
    y = np.sin(longitudes) * np.cos(latitudes)
    z = np.sin(latitudes)
    return x, y, z


def radians(angles):
    """Radian converter

    radians = degrees * (pi / 180)

    :param angles: array of angles specified in degrees
    :returns: array of angles converted to radians
    """
    return np.asarray(angles, dtype="d") * (np.pi / 180.)
