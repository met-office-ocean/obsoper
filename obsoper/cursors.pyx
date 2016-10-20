"""
Methods for navigating ocean grids
"""
from collections import defaultdict
from . import orca


cdef class Cursor:
    """Regular grid cursor definition"""
    def __init__(self, int ni, int nj):
        self.ni = ni
        self.nj = nj

    @classmethod
    def tripolar(cls, grid_longitudes, grid_latitudes, fold_index=-1):
        """Tri-polar grid cursor"""
        ni, nj = grid_longitudes.shape
        north_fold = orca.north_fold(grid_longitudes[:, fold_index],
                                     grid_latitudes[:, fold_index])
        return Tripolar(ni, nj, north_fold)

    cpdef tuple move(self, int i, int j, int di, int dj):
        """Basic movement instruction"""
        return i + di, j + dj


cdef class Tripolar(Cursor):
    """Tri-polar grid cursor definition"""
    def __init__(self, int ni, int nj, dict fold):
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
