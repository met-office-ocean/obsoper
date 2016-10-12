cdef class Box:
    cdef:
        double xmin, xmax, ymin, ymax
    cpdef bint inside(self, double x, double y)

cdef class Dateline(Box):
    cdef:
        double north
        double south
        double east
        double west
    cpdef bint inside(self, double longitude, double latitude)
    cpdef bint inside_x(self, double longitude)
    cpdef bint inside_y(self, double latitude)
