cdef class Cursor:
    cdef int ni, nj
    cpdef tuple move(self, int i, int j, int di, int dj)


cdef class Tripolar(Cursor):
    cdef NorthFold fold
    cpdef tuple move(self, int i, int j, int di, int dj)
    cpdef tuple index(self, int i, int j)
    cpdef bint on_fold(self, int j)
    cpdef tuple map_fold(self, int i)


cdef class NorthFold:
    cdef dict northfold_map
