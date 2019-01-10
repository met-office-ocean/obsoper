import numpy as np


def select_field(field, i, j):
    """Select array of grid cell corner values"""
    return select_corners(field, i, j)


# Vectorised algorithm
def select_corners(values, i, j):
    """Select array of grid cell corner positions/columns

    Selects positions/water columns representing four corners by
    choosing (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)

    .. note:: periodic in i coordinate

    :param values: 2D/3D array dimension (X, Y, [Z])
    :param i: array of integers shaped ([N],) representing lower left corner
    :param j: array of integers shaped ([N],) representing lower left corner
    :returns: array shaped (4, [Z, [N]]) representing values/columns
              for each cell whose lower left corner is (i, j)
    """
    #pylint: disable=invalid-name
    i, j = np.asarray(i, dtype="i"), np.asarray(j, dtype="i")

    if isinstance(values, np.ma.MaskedArray):
        array_constructor = np.ma.MaskedArray
    else:
        array_constructor = np.array
    ni = values.shape[0]
    vectors = [values[i % ni, j],
               values[(i+1) % ni, j],
               values[(i+1) % ni, j+1],
               values[i % ni, j+1]]
    if np.ndim(i) == 0:
        return array_constructor(vectors, dtype="d")
    else:
        return array_constructor([vector.T for vector in vectors], dtype="d")


def correct_corners(vertices, longitudes):
    """Alter grid corners in-place to enable vectorised interpolation"""
    vertices = np.asarray(vertices, dtype="d")
    grid_longitudes = vertices[:, 0]

    # Apply East/West dateline corrections
    if vertices.ndim == 2:
        if is_dateline(vertices):
            if is_east(longitudes):
                correct_east(grid_longitudes)
            else:
                correct_west(grid_longitudes)

    elif vertices.ndim == 3:
        datelines = is_dateline(vertices)

        east = is_east(longitudes) & datelines
        west = is_west(longitudes) & datelines

        # Note: boolean indexing returns copy of array
        grid_longitudes[:, east] = correct_east(grid_longitudes[:, east])
        grid_longitudes[:, west] = correct_west(grid_longitudes[:, west])

    return vertices


def correct_east(longitudes):
    """Map cell lying on 180th meridian to eastern coordinates"""
    longitudes[longitudes > 0] -= 360.
    return longitudes


def correct_west(longitudes):
    """Map cell lying on 180th meridian to western coordinates"""
    longitudes[longitudes < 0] += 360.
    return longitudes


def is_dateline(vertices):
    """Detect dateline cells from corners"""
    vertices = np.asarray(vertices, dtype="d")
    longitudes = vertices[:, 0]
    return np.abs(longitudes.min(axis=0) - longitudes.max(axis=0)) > 180


def is_east(longitudes):
    """Detect positions on eastern side of greenwich meridian"""
    return np.asarray(longitudes, dtype="d") <= 0.


def is_west(longitudes):
    """Detect positions on western side of greenwich meridian"""
    return ~is_east(longitudes)


def mask_corners(corner_values):
    """Masks cells which have 1 or more masked corner values

    :param corner_values: array shaped ([[N], M], 4)
    :returns: array with corner dimension partially masked cells fully masked
    """
    if hasattr(corner_values, "mask") and (corner_values.mask.any()):
        if corner_values.ndim == 1:
            return np.ma.masked_all_like(corner_values)

        # Multidimensional arrays
        invalid = corner_values.mask.any(axis=-1)
        corner_values[invalid] = np.ma.masked
    return corner_values
