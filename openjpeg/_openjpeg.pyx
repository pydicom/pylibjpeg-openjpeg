# cython: language_level=3
# distutils: language=c
from math import ceil
from io import BytesIO
import logging
from typing import Union, Dict, BinaryIO, Tuple, List

from libc.stdint cimport uint32_t

from cpython.ref cimport PyObject
import numpy as np
cimport numpy as cnp

cdef extern struct JPEG2000Parameters:
    uint32_t columns
    uint32_t rows
    int colourspace
    uint32_t samples_per_pixel
    uint32_t precision
    unsigned int is_signed
    uint32_t nr_tiles

cdef extern char* OpenJpegVersion()
cdef extern int Decode(void* fp, unsigned char* out, int codec)
cdef extern int GetParameters(void* fp, int codec, JPEG2000Parameters *param)
cdef extern int EncodeArray(
    cnp.PyArrayObject* arr,
    PyObject* dst,
    int bits_stored,
    int photometric_interpretation,
    bint use_mct,
    PyObject* compression_ratios,
    PyObject* signal_noise_ratios,
    int codec_format,
)
cdef extern int EncodeBuffer(
    PyObject* src,
    int columns,
    int rows,
    int samples_per_pixel,
    int bits_stored,
    int is_signed,
    int photometric_interpretation,
    PyObject* dst,
    bint use_mct,
    PyObject* compression_ratios,
    PyObject* signal_noise_ratios,
    int codec_format,
)


LOGGER = logging.getLogger(__name__)
ERRORS = {
    1: "failed to create the input stream",
    2: "failed to setup the decoder",
    3: "failed to read the header",
    4: "failed to set the component indices",
    5: "failed to set the decoded area",
    6: "failed to decode image",
    7: "support for more than 32-bits per component is not implemented",
    8: "failed to upscale subsampled components",
}


def get_version() -> bytes:
    """Return the openjpeg version as bytes."""
    cdef char *version = OpenJpegVersion()

    return version


def decode(
    fp: BinaryIO,
    codec: int = 0,
    as_array: bool = False
) -> Union[np.ndarray, bytearray]:
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
    as_array : bool, optional
        If ``True`` then return the decoded image data as a :class:`numpy.ndarray`
        otherwise return the data as a :class:`bytearray` (default).

    Returns
    -------
    bytearray | numpy.ndarray
        If `as_array` is False (default) then returns the decoded image data
        as a :class:`bytearray`, otherwise returns the image data as a
        :class:`numpy.ndarray`.

    Raises
    ------
    RuntimeError
        If unable to decode the JPEG 2000 data.
    """
    param = get_parameters(fp, codec)
    bpp = ceil(param['precision'] / 8)
    if bpp == 3:
        bpp = 4
    nr_bytes = param['rows'] * param['columns'] * param['samples_per_pixel'] * bpp

    cdef PyObject* p_in = <PyObject*>fp
    cdef unsigned char *p_out
    if as_array:
        out = np.zeros(nr_bytes, dtype=np.uint8)
        p_out = <unsigned char *>cnp.PyArray_DATA(out)
    else:
        out = bytearray(nr_bytes)
        p_out = <unsigned char *>out

    return_code = Decode(p_in, p_out, codec)

    return return_code, out


def get_parameters(fp: BinaryIO, codec: int = 0) -> Dict[str, Union[str, int, bool]]:
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
        'samples_per_pixel: int, 'precision': int, `is_signed`: bool,
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
    param.samples_per_pixel = 0
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
        'samples_per_pixel' : param.samples_per_pixel,
        'precision' : param.precision,
        'is_signed' : bool(param.is_signed),
        'nr_tiles' : param.nr_tiles,
    }

    return parameters


def encode_array(
    cnp.ndarray arr,
    int bits_stored,
    int photometric_interpretation,
    bint use_mct,
    List[float] compression_ratios,
    List[float] signal_noise_ratios,
    int codec_format,
) -> Tuple[int, bytes]:
    """Return the JPEG 2000 compressed `arr`.

    Parameters
    ----------
    arr : numpy.ndarray
        The array containing the image data to be encoded.
    bits_stored : int, optional
        The number of bits used per pixel.
    photometric_interpretation : int
        The colour space of the unencoded image data that will be set in the
        JP2 metadata (if `codec_format` is ``1``).
    use_mct : bool
        If ``True`` then apply multi-component transformation (MCT) to RGB
        images.
    compression_ratios : list[float]
        Required for lossy encoding, this is the compression ratio to use
        for each quality layer. Cannot be used with `signal_noise_ratios`.
    signal_noise_ratios : list[float]
        Required for lossy encoding, this is the PSNR to use for each quality
        layer. Cannot be used with `compression_ratios`.
    codec_format : int
        The codec to used when encoding:

        * ``0``: JPEG 2000 codestream only (default) (J2K/J2C format)
        * ``1``: A boxed JPEG 2000 codestream (JP2 format)

    Returns
    -------
    tuple[int, bytes]
        The return code of the encoding and the encoded image data. The return
        code will be ``0`` for success, otherwise the encoding failed.
    """
    if not (1 <= bits_stored <= arr.dtype.itemsize * 8):
        raise ValueError(
            "Invalid value for the 'bits_stored' parameter, the value must be "
            f"in the range (1, {arr.dtype.itemsize * 8})"
        )

    kind, itemsize = arr.dtype.kind, arr.dtype.itemsize
    if kind not in ("b", "i", "u") or itemsize not in (1, 2, 4):
        raise ValueError(
            f"The input array has an unsupported dtype '{arr.dtype}', only bool, "
            "u1, u2, u4, i1, i2 and i4 are supported"
        )

    # It seems like OpenJPEG can only encode up to 24 bits, although theoretically
    #   based on their use of OPJ_INT32 for pixel values, it should be 32-bit for
    #   signed and 31 bit for unsigned. Maybe I've made a mistake somewhere?
    arr_max = arr.max()
    arr_min = arr.min()
    if (
        (kind == "u" and itemsize == 4 and arr_max > 2**24 - 1)
        or (kind == "i" and itemsize == 4 and (arr_max > 2**23 - 1 or arr_min < -2**23))
    ):
        raise ValueError(
            "The input array contains values outside the range of the maximum "
            "supported bit-depth of 24"
        )

    # Check the array matches bits_stored
    if kind == "u" and itemsize in (1, 2, 4) and arr_max > 2**bits_stored - 1:
        raise ValueError(
            f"A 'bits_stored' value of {bits_stored} is incompatible with "
            f"the range of pixel data in the input array: ({arr_min}, {arr_max})"
        )

    if (
        kind == "i" and itemsize in (1, 2, 4)
        and (arr_max > 2**(bits_stored - 1) - 1 or arr_min < -2**(bits_stored - 1))
    ):
        raise ValueError(
            f"A 'bits_stored' value of {bits_stored} is incompatible with "
            f"the range of pixel data in the input array: ({arr_min}, {arr_max})"
        )

    # MCT may be used with RGB in both lossy and lossless modes
    use_mct = 1 if use_mct else 0

    if codec_format not in (0, 1):
        raise ValueError(
            f"Invalid 'codec_format' value '{codec_format}', must be 0 or 1"
        )

    compression_ratios = [float(x) for x in compression_ratios]
    signal_noise_ratios = [float(x) for x in signal_noise_ratios]
    if compression_ratios and signal_noise_ratios:
        raise ValueError(
            "Only one of 'compression_ratios' or 'signal_noise_ratios' is "
            "allowed when performing lossy compression"
        )
    if len(compression_ratios) > 10 or len(signal_noise_ratios) > 10:
        raise ValueError("More than 10 compression layers is not supported")

    # The destination for the encoded J2K codestream, needs to support BinaryIO
    dst = BytesIO()
    return_code = EncodeArray(
        <cnp.PyArrayObject *> arr,
        <PyObject *> dst,
        bits_stored,
        photometric_interpretation,
        use_mct,
        <PyObject *> compression_ratios,
        <PyObject *> signal_noise_ratios,
        codec_format,
    )
    return return_code, dst.getvalue()


def encode_buffer(
    src,
    int columns,
    int rows,
    int samples_per_pixel,
    int bits_stored,
    int is_signed,
    int photometric_interpretation,
    int use_mct,
    List[float] compression_ratios,
    List[float] signal_noise_ratios,
    int codec_format,
) -> Tuple[int, bytes]:
    """Return the JPEG 2000 compressed `src`.

    If performing lossy encoding then either `compression_ratios` or
    `signal_noise_ratios` must be set to a non-empty list, otherwise lossless
    encoding will be used.

    Parameters
    ----------
    src : bytes | bytearray
        A bytes or bytearray containing the image data to be encoded, ordered as
        little endian and colour-by-pixel.
    columns : int
        The number of columns in the image, should be in the range [1, 65535].
    rows : int
        The number of rows in the image, should be in the range [1, 65535].
    samples_per_pixel : int
        The number of samples per pixel, should be 1, 3 or 4.
    bits_stored : int
        The number of bits used per pixel (i.e. the sample precision), should be
        in the range [1, 24].
    is_signed: int
        ``0`` if the image uses unsigned pixels, ``1`` for signed.
    photometric_interpretation : int
        The colour space of the unencoded image data that will be set in the
        JP2 metadata (if `codec_format` is ``1``).
    use_mct : bool
        If ``1`` then apply multi-component transformation (MCT) to RGB
        images. Requires a `photometric_interpretation` of ``1`` and a
        `samples_per_pixel` value of ``3``, otherwise no MCT will be used.
    compression_ratios : list[float]
        Required for lossy encoding, this is the compression ratio to use
        for each quality layer. Cannot be used with `signal_noise_ratios`.
    signal_noise_ratios : list[float]
        Required for lossy encoding, this is the PSNR to use for each quality
        layer. Cannot be used with `compression_ratios`.
    codec_format : int, optional
        The codec to used when encoding:

        * ``0``: JPEG 2000 codestream only (default) (J2K/J2C format)
        * ``1``: A boxed JPEG 2000 codestream (JP2 format)

    Returns
    -------
    tuple[int, bytes]
        The return code of the encoding and the JPEG 2000 encoded image data.
        The return code will be ``0`` for success, otherwise the encoding
        failed.
    """
    # Checks
    if not isinstance(src, (bytes, bytearray)):
        raise TypeError(
            f"'src' must be bytes or bytearray, not {type(src).__name__}"
        )

    if not 1 <= columns <= 65535:
        raise ValueError(
            f"Invalid 'columns' value '{columns}', must be in the range [1, 65535]"
        )

    if not 1 <= rows <= 65535:
        raise ValueError(
            f"Invalid 'rows' value '{rows}', must be in the range [1, 65535]"
        )

    if samples_per_pixel not in (1, 3, 4):
        raise ValueError(
            f"Invalid 'samples_per_pixel' value '{samples_per_pixel}', must be 1, 3 "
            "or 4"
        )

    if 0 < bits_stored <= 8:
        bytes_allocated = 1
    elif 8 < bits_stored <= 16:
        bytes_allocated = 2
    elif 16 < bits_stored <= 24:
        bytes_allocated = 4
    else:
        raise ValueError(
            f"Invalid 'bits_stored' value '{bits_stored}', must be in the "
            "range [1, 24]"
        )

    actual_length = len(src)
    expected_length = rows * columns * samples_per_pixel * bytes_allocated
    if actual_length != expected_length:
        raise ValueError(
            f"The length of 'src' is {actual_length} bytes which doesn't "
            f"match the expected length of {expected_length} bytes"
        )

    if is_signed not in (0, 1):
        raise ValueError(f"Invalid 'is_signed' value '{is_signed}'")

    if photometric_interpretation not in (0, 1, 2, 3, 4, 5):
        raise ValueError(
            "Invalid 'photometric_interpretation' value "
            f"'{photometric_interpretation}', must be in the range [0, 5]"
        )

    if use_mct not in (0, 1):
        raise ValueError(f"Invalid 'use_mct' value '{use_mct}'")

    if codec_format not in (0, 1):
        raise ValueError(
            f"Invalid 'codec_format' value '{codec_format}', must be 0 or 1"
        )

    compression_ratios = [float(x) for x in compression_ratios]
    signal_noise_ratios = [float(x) for x in signal_noise_ratios]
    if compression_ratios and signal_noise_ratios:
        raise ValueError(
            "Only one of 'compression_ratios' or 'signal_noise_ratios' is "
            "allowed when performing lossy compression"
        )
    if len(compression_ratios) > 10 or len(signal_noise_ratios) > 10:
        raise ValueError("More than 10 compression layers is not supported")

    dst = BytesIO()
    return_code = EncodeBuffer(
        <PyObject *> src,
        columns,
        rows,
        samples_per_pixel,
        bits_stored,
        is_signed,
        photometric_interpretation,
        <PyObject *> dst,
        use_mct,
        <PyObject *> compression_ratios,
        <PyObject *> signal_noise_ratios,
        codec_format,
    )
    return return_code, dst.getvalue()
