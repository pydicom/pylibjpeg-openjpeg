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
cdef extern int decode(void* fp, unsigned char* out)
cdef extern int read_data(PyObject* fp, char* destination, int nr_bytes)


def get_version():
    """Return the openjpeg version as bytes."""
    cdef char *version = OpenJpegVersion()

    return version

def opj_decode(fp, nr_bytes):
    cdef PyObject* ptr = <PyObject*>fp
    output_buffer = np.zeros(nr_bytes, dtype=np.uint8)
    cdef unsigned char *p_out = <unsigned char *>np.PyArray_DATA(output_buffer)

    cdef int result = decode(ptr, p_out)

    return result, output_buffer
