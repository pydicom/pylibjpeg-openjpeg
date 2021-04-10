# cython: language_level=3
# distutils: language=c
from math import ceil

from libc.stdint cimport uint32_t

from cpython.ref cimport PyObject
import numpy as np
cimport numpy as np

cdef extern struct JPEG2000Parameters:
    uint32_t columns
    uint32_t rows
    int colourspace
    uint32_t nr_components
    uint32_t precision
    unsigned int is_signed
    uint32_t nr_tiles

cdef extern char* OpenJpegVersion()
cdef extern int Decode(void* fp, unsigned char* out, int codec)
cdef extern int GetParameters(void* fp, int codec, JPEG2000Parameters *param)


ERRORS = {
    1: "failed to create the input stream",
    2: "failed to setup the decoder",
    3: "failed to read the header",
    4: "failed to set the component indices",
    5: "failed to set the decoded area",
    6: "failed to decode image",
    7: "support for more than 16-bits per component is not implemented",
    8: "failed to upscale subsampled components",
}


def get_version():
    """Return the openjpeg version as bytes."""
    cdef char *version = OpenJpegVersion()

    return version


def decode(fp, codec=0):
    """Return the decoded JPEG 2000 data from Python file-like `fp`.

    Parameters
    ----------
    fp : file-like
        A Python file-like containing the encoded JPEG 2000 data. Must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    codec : int, optional
        The codec to use for decoding, one of:

        * ``0``: JPEG-2000 codestream
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format

    Returns
    -------
    numpy.ndarray
        An ndarray of uint8 containing the decoded image data.

    Raises
    ------
    RuntimeError
        If unable to decode the JPEG 2000 data.
    """
    param = get_parameters(fp, codec)
    bpp = ceil(param['precision'] / 8)
    nr_bytes = param['rows'] * param['columns'] * param['nr_components'] * bpp

    cdef PyObject* p_in = <PyObject*>fp
    arr = np.zeros(nr_bytes, dtype=np.uint8)
    cdef unsigned char *p_out = <unsigned char *>np.PyArray_DATA(arr)

    result = Decode(p_in, p_out, codec)
    if result != 0:
        try:
            msg = f": {ERRORS[result]}"
        except KeyError:
            pass

        raise RuntimeError("Error decoding the J2K data" + msg)

    return arr


def get_parameters(fp, codec=0):
    """Return a :class:`dict` containing the JPEG 2000 image parameters.

    Parameters
    ----------
    fp : file-like
        A Python file-like containing the encoded JPEG 2000 data.
    codec : int, optional
        The codec to use for decoding, one of:

        * ``0``: JPEG-2000 codestream
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format

    Returns
    -------
    dict
        A :class:`dict` containing the J2K image parameters:
        ``{'columns': int, 'rows': int, 'colourspace': str,
        'nr_components: int, 'precision': int, `is_signed`: bool,
        'nr_tiles: int'}``. Possible colour spaces are "unknown",
        "unspecified", "sRGB", "monochrome", "YUV", "e-YCC" and "CYMK".

    Raises
    ------
    RuntimeError
        If unable to decode the JPEG 2000 data.
    """
    cdef JPEG2000Parameters param
    param.columns = 0
    param.rows = 0
    param.colourspace = 0
    param.nr_components = 0
    param.precision = 0
    param.is_signed = 0
    param.nr_tiles = 0

    # Pointer to the JPEGParameters object
    cdef JPEG2000Parameters *p_param = &param

    # Pointer to J2K data
    cdef PyObject* ptr = <PyObject*>fp

    # Decode the data - output is written to output_buffer
    result = GetParameters(ptr, codec, p_param)
    if result != 0:
        try:
            msg = f": {ERRORS[result]}"
        except KeyError:
            pass

        raise RuntimeError("Error decoding the J2K data" + msg)

    # From openjpeg.h#L309
    colours = {
        -1: "unknown",
         0: "unspecified",
         1: "sRGB",
         2: "monochrome",
         3: "YUV",
         4: "e-YCC",
         5: "CYMK",
    }

    try:
        colourspace = colours[param.colourspace]
    except KeyError:
        colourspace = "unknown"

    parameters = {
        'rows' : param.rows,
        'columns' : param.columns,
        'colourspace' : colourspace,
        'nr_components' : param.nr_components,
        'precision' : param.precision,
        'is_signed' : bool(param.is_signed),
        'nr_tiles' : param.nr_tiles,
    }

    return parameters
