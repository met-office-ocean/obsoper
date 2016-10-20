"""
ORCA grid specific methods
"""
from collections import defaultdict


def north_fold(longitudes, latitudes):
    """Northern hemisphere tri-polar fold"""
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
