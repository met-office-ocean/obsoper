"""
Grid walk algorithm
"""
# pylint: disable=invalid-name
import numpy as np
from . import exceptions
from . import (cursors,
               cell)
from . cimport (cursors,
                spherical,
                cell)
cimport cython


cdef class Walk:
    """Walk a grid to find cell containing point

    Iterates through steps in i, j coordinates by intersecting a line that
    joins the cell center to the traget point with an edge of the cell.
    """
    cdef:
        cell.Collection cells
        cursors.Cursor cursor
        int ni, nj, max_steps

    extra_steps = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(self, cell.Collection cells, cursors.Cursor cursor):
        self.cells = cells
        self.cursor = cursor
        self.ni, self.nj = self.cells.shape
        self.max_steps = self.ni * self.nj

    @classmethod
    def tripolar(cls, grid_longitudes, grid_latitudes, fold_index=-1):
        """Factory method"""
        cdef:
            cursors.Cursor cursor
            cell.Collection cells

        cursor = cursors.Cursor.tripolar(grid_longitudes,
                                         grid_latitudes,
                                         fold_index)
        cells = cell.Collection.from_lonlats(grid_longitudes,
                                             grid_latitudes)
        return cls(cells, cursor)

    @classmethod
    def from_lonlats(cls, grid_longitudes, grid_latitudes):
        """Factory method"""
        ni, nj = grid_longitudes.shape
        cursor = cursors.Cursor(ni, nj)
        cells = cell.Collection.from_lonlats(grid_longitudes,
                                             grid_latitudes)
        return cls(cells, cursor)

    def query(self, longitudes, latitudes, igrid, jgrid):
        """Find multiple grid points by walking from starting points"""
        return self._query(np.asarray(longitudes, dtype="d"),
                           np.asarray(latitudes, dtype="d"),
                           np.asarray(igrid, dtype="i"),
                           np.asarray(jgrid, dtype="i"))

    @cython.boundscheck(False)
    cdef tuple _query(self,
                      double[:] longitudes,
                      double[:] latitudes,
                      int[:] igrid,
                      int[:] jgrid):
        """Find multiple grid points by walking from starting points"""
        cdef:
            int ipoint
            int npoints = longitudes.shape[0]
            int[:] ifinal = np.zeros(npoints, dtype="i")
            int[:] jfinal = np.zeros(npoints, dtype="i")

        for ipoint in range(npoints):
            ifinal[ipoint], jfinal[ipoint] = self._query_one(longitudes[ipoint],
                                                             latitudes[ipoint],
                                                             igrid[ipoint],
                                                             jgrid[ipoint])
        return ifinal, jfinal

    def query_one(self, point, int i=0, int j=0):
        """find point in grid"""
        cdef double longitude, latitude
        longitude = point[0]
        latitude = point[1]
        return self._query_one(longitude, latitude, i, j)

    def _query_one(self, double longitude, double latitude, int i=0, int j=0):
        """find point in grid"""
        i, j = self.adjust_start(i, j)
        return self.detect((longitude, latitude), i, j)

    def detect(self, point, int i=0, int j=0):
        """Detect Cell containing point"""
        cdef:
            int istep = 0
            int di, dj
            double x, y
            cell.Cell grid_cell

        while istep < self.max_steps:
            # Check candidate cell for point membership
            grid_cell = self.cells.find(i, j)
            x, y = point
            if grid_cell.contains(x, y):
                return i, j

            try:
                di, dj = self.direction(grid_cell, point)
            except exceptions.StepNotFound:
                return self.brute_force(i, j, point)

            # Move cursor to next cell
            i, j = self.cursor.move(i, j, di, dj)

            istep += 1

        # Handle case where valid cell can not be detected
        message = ("Walk exceeded max. iterations: {} {} {}").format(point,
                                                                     i, j)
        raise Exception(message)

    def brute_force(self, int i, int j, point):
        """Brute force search of four nearest grid boxes"""
        x, y = point
        for di, dj in self.extra_steps:
            inear, jnear = self.cursor.move(i, j, di, dj)
            grid_cell = self.cells.find(inear, jnear)
            if grid_cell.contains(x, y):
                return inear, jnear
        message = ("Radial algorithm exhausted list of steps: "
                   "({}, {}) ({}, {})").format(i, j, x, y)
        raise Exception(message)

    cpdef direction(self, cell.Cell grid_cell, point):
        """Calculate grid step from grid location and point"""
        return next_step(grid_cell.polygon, grid_cell.center, point)

    def adjust_start(self, int i, int j):
        """Handle initial points on grid boundary"""
        if i == (self.ni - 1):
            i -= 1
        if j == (self.nj - 1):
            j -= 1
        return i, j


cpdef tuple next_step(double[:, :] vertices, center, point):
    """Estimate next direction to step across the grid"""
    cdef double[:, :] connecting_line = np.array([center, point],
                                                 dtype=np.double)
    cdef int nlines = 4
    cdef int iline = 0
    cdef list steps = [(0, -1),
                       (1, 0),
                       (0, 1),
                       (-1, 0)]

    cdef double[:, :] edge = np.empty((2, 2), dtype=np.double)
    for iline in range(nlines):
        edge[0, :] = vertices[iline, :]
        edge[1, :] = vertices[(iline + 1) % nlines, :]
        if spherical.intersect(edge, connecting_line):
            return steps[iline]

    raise exceptions.StepNotFound("Step not found")
