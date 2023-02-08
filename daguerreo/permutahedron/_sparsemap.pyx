# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from lpsmap.ad3qp.base cimport PGenericFactor


cdef class Permutahedron(PGenericFactor):

    def __cinit__(self, bool allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new _Permutahedron()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int length):
        (<_Permutahedron*>self.thisptr).Initialize(length)

    # def print(self):
        # (<GenericFactor*>self.thisptr).PrintActiveSet()
