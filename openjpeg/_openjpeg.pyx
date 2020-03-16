# cython: language_level=3
# distutils: language=c

#from math import ceil

#from libcpp cimport bool
#from libcpp.string cimport string

#import numpy as np
#cimport numpy as np


#cdef extern from "Jpeg2KDecode.c":
#cdef extern from "utils.c":
cdef extern char* OpenJpegVersion()



def get_version():
    """Return the openjpeg version."""
    cdef char *version = OpenJpegVersion()

    return version
