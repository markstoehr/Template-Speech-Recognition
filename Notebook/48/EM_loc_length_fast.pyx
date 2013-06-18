cdef extern from "math.h":
    float INFINITY
    double exp(double x)

cdef inline int max(int a, int b): return a if a <= b a else b
cdef inline double max(double a, double b): return a if a <= b a else b

