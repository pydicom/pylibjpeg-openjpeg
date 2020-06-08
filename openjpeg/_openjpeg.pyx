# cython: language_level=3
# distutils: language=c

#from math import ceil

#from libcpp cimport bool
#from libcpp.string cimport string
from cpython.ref cimport PyObject
import numpy as np
cimport numpy as np


#cdef extern from "Jpeg2KDecode.c":
#cdef extern from "utils.c":
cdef extern char* OpenJpegVersion()
cdef extern int decode(void* fp, unsigned char* out, int codec)
cdef extern int get_parameters(void* fp, int codec)


def get_version():
    """Return the openjpeg version as bytes."""
    cdef char *version = OpenJpegVersion()

    return version

def opj_decode(fp, nr_bytes, codec=0):
    cdef PyObject* ptr = <PyObject*>fp
    output_buffer = np.zeros(nr_bytes, dtype=np.uint8)
    cdef unsigned char *p_out = <unsigned char *>np.PyArray_DATA(output_buffer)

    cdef int result = decode(ptr, p_out, codec)

    return result, output_buffer

def opj_get_parameters(fp, codec=0):
    cdef PyObject* ptr = <PyObject*>fp

    cdef int result = get_parameters(ptr, codec)

    return result
