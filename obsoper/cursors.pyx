"""
Methods for navigating ocean grids
"""
from collections import defaultdict


cdef class Cursor:
    """Regular grid cursor definition"""
    def __init__(self, int ni, int nj):
        self.ni = ni
        self.nj = nj

    @classmethod
    def tripolar(cls, grid_longitudes, grid_latitudes, fold_index=-1):
        """Tri-polar grid cursor"""
        ni, nj = grid_longitudes.shape
        north_fold = NorthFold(grid_longitudes[:, fold_index],
                               grid_latitudes[:, fold_index])
        return Tripolar(ni, nj, north_fold)

    cpdef tuple move(self, int i, int j, int di, int dj):
        """Basic movement instruction"""
        return i + di, j + dj


cdef class Tripolar(Cursor):
    """Tri-polar grid cursor definition"""
    def __init__(self, int ni, int nj, NorthFold fold):
        self.ni = ni
        self.nj = nj
        self.fold = fold

    cpdef tuple move(self, int i, int j, int di, int dj):
        """Basic movement instruction"""
        return self.index(i + di, j + dj)

    cpdef tuple index(self, int i, int j):
        """Dereference cyclic tripolar grid indices to array indices"""
        if self.on_fold(j):
            i, j = self.map_fold(i)
        return i % self.ni, j

    cpdef bint on_fold(self, int j):
        """Detect move which lands on north fold"""
        return j >= (self.nj - 1)

    cpdef tuple map_fold(self, int i):
        """Apply north fold i-coordinate mapping"""
        return self.fold[i], self.nj - 2


cdef class NorthFold:
    """Nothern hemisphere tri-polar grid fold.

    The ORCA family of grids include a row of positions that
    represent the north fold.
    """

    def __init__(self, longitudes, latitudes):
        # Match indices to coordinates
        coordinates = defaultdict(list)
        for ikey, key in enumerate(zip(longitudes, latitudes)):
            coordinates[key].append(ikey)

        # Create bijective map between north fold indices
        self.northfold_map = {}
        for indices in coordinates.itervalues():
            if len(indices) == 2:
                j1, j2 = indices
                self.northfold_map[j1] = j2
                self.northfold_map[j2] = j1

    def __getitem__(self, index):
        """translate position in grid to opposite side of north fold"""
        return self.northfold_map[index]
