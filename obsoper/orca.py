"""
ORCA grid specific methods
"""
from collections import defaultdict
import numpy as np


def north_fold(longitudes, latitudes):
    """Northern hemisphere tri-polar fold

    :param longitudes: 1D array representing longitudes on north fold
    :param latitudes: 1D array representing latitudes on north fold
    :returns: mapping between folded edges
    """
    # Match indices to coordinates
    coordinates = defaultdict(list)
    for ikey, key in enumerate(zip(longitudes, latitudes)):
        coordinates[key].append(ikey)

    # Create bijective map between north fold indices
    result = {}
    for indices in coordinates.itervalues():
        if len(indices) == 2:
            j1, j2 = indices
            result[j1] = j2
            result[j2] = j1
    return result


def remove_halo(field):
    """Removes extra row and columns used by NEMO models

    :param field: 2D array dimensioned (x, y)
    """
    if not isinstance(field, np.ndarray):
        field = np.asarray(field)
    if len(field) <= 2:
        return np.array([])
    return field[1:-1, :-1]
