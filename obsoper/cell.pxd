from . cimport box

cdef class Cell:
    cdef:
        public double[:, :] vertices
        public box.Box bounding_box
        tuple _center
    cpdef bint contains(self, double x, double y)
    cdef bint same_side_test(self, double x, double y)
    cdef tuple _compute_center(self)

cdef class Collection:
    cdef:
        double[:, :, :] positions
        dict cache
    cpdef Cell find(self, int i, int j)
