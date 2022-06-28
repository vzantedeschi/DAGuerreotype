# cython: language_level=3
# distutils: language=c++

from lpsmap.ad3qp.base cimport GenericFactor


cdef extern from "_permutahedron_src.hpp" namespace "AD3":

    cdef cppclass _Permutahedron(GenericFactor):
        _Permutahedron()
        void Initialize(int length)
