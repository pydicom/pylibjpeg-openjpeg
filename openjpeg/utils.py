from enum import IntEnum
from io import BytesIO
from math import ceil
import os
from pathlib import Path
from typing import BinaryIO, Tuple, Union, TYPE_CHECKING, Any, Dict, cast
import warnings

import numpy as np

import _openjpeg


if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


class Version(IntEnum):
    v1 = 1
    v2 = 2


MAGIC_NUMBERS = {
    # JPEG 2000 codestream, has no header, .j2k, .jpc, .j2c
    b"\xff\x4f\xff\x51": 0,
    # JP2 and JP2 RFC3745, .jp2
    b"\x0d\x0a\x87\x0a": 2,
    b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a": 2,
    # JPT, .jpt - shrug
}


def _get_format(stream: BinaryIO) -> int:
    """Return the JPEG 2000 format for the encoded data in `stream`.

    Parameters
    ----------
    stream : file-like
        A Python object containing the encoded JPEG 2000 data. If not
        :class:`bytes` then the object must have ``tell()``, ``seek()`` and
        ``read()`` methods.

    Returns
    -------
    int
        The format of the encoded data, one of:

        * ``0``: JPEG-2000 codestream
        * ``2``: JP2 file format

    Raises
    ------
    ValueError
        If no matching JPEG 2000 file format found for the data.
    """
    data = stream.read(20)
    stream.seek(0)

    try:
        return MAGIC_NUMBERS[data[:4]]
    except KeyError:
        pass

    try:
        return MAGIC_NUMBERS[data[:12]]
    except KeyError:
        pass

    raise ValueError("No matching JPEG 2000 format found")


def get_openjpeg_version() -> Tuple[int, ...]:
    """Return the openjpeg version as tuple of int."""
    version = _openjpeg.get_version().decode("ascii").split(".")
    return tuple([int(ii) for ii in version])


def decode(
    stream: Union[str, os.PathLike, bytes, bytearray, BinaryIO],
    j2k_format: Union[int, None] = None,
    reshape: bool = True,
) -> np.ndarray:
    """Return the decoded JPEG2000 data from `stream` as a
    :class:`numpy.ndarray`.

    .. versionchanged:: 1.1

        `stream` can now also be :class:`str` or :class:`pathlib.Path`

    Parameters
    ----------
    stream : str, pathlib.Path, bytes or file-like
        The path to the JPEG 2000 file or a Python object containing the
        encoded JPEG 2000 data. If using a file-like then the object must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    j2k_format : int, optional
        The JPEG 2000 format to use for decoding, one of:

        * ``0``: JPEG-2000 codestream (such as from DICOM *Pixel Data*)
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format
    reshape : bool, optional
        Reshape and re-view the output array so it matches the image data
        (default), otherwise return a 1D array of ``np.uint8``.

    Returns
    -------
    numpy.ndarray
        An array of containing the decoded image data.

    Raises
    ------
    RuntimeError
        If the decoding failed.
    """
    if isinstance(stream, (str, Path)):
        with open(stream, "rb") as f:
            buffer: BinaryIO = BytesIO(f.read())
            buffer.seek(0)
    elif isinstance(stream, (bytes, bytearray)):
        buffer = BytesIO(stream)
    else:
        # BinaryIO
        required_methods = ["read", "tell", "seek"]
        if not all([hasattr(stream, meth) for meth in required_methods]):
            raise TypeError(
                "The Python object containing the encoded JPEG 2000 data must "
                "either be bytes or have read(), tell() and seek() methods."
            )
        buffer = cast(BinaryIO, stream)

    if j2k_format is None:
        j2k_format = _get_format(buffer)

    if j2k_format not in [0, 1, 2]:
        raise ValueError(f"Unsupported 'j2k_format' value: {j2k_format}")

    arr = cast(np.ndarray, _openjpeg.decode(buffer, j2k_format, as_array=True))
    if not reshape:
        return arr

    meta = get_parameters(buffer, j2k_format)
    precision = cast(int, meta["precision"])
    rows = cast(int, meta["rows"])
    columns = cast(int, meta["columns"])
    pixels_per_sample = cast(int, meta["nr_components"])
    pixel_representation = cast(bool, meta["is_signed"])
    bpp = ceil(precision / 8)

    dtype = f"u{bpp}" if not pixel_representation else f"i{bpp}"
    arr = arr.view(dtype)

    shape = [rows, columns]
    if pixels_per_sample > 1:
        shape.append(pixels_per_sample)

    return arr.reshape(*shape)


def decode_pixel_data(
    src: bytes,
    ds: Union["Dataset", Dict[str, Any], None] = None,
    version: int = Version.v1,
    **kwargs: Any,
) -> Union[np.ndarray, bytearray]:
    """Return the decoded JPEG 2000 data as a :class:`numpy.ndarray`.

    Intended for use with *pydicom* ``Dataset`` objects.

    Parameters
    ----------
    src : bytes
        A Python object containing the encoded JPEG 2000 data. If not
        :class:`bytes` then the object must have ``tell()``, ``seek()`` and
        ``read()`` methods.
    ds : pydicom.dataset.Dataset, optional
        A :class:`~pydicom.dataset.Dataset` containing the group ``0x0028``
        elements corresponding to the *Pixel data*. If used then the
        *Samples per Pixel*, *Bits Stored* and *Pixel Representation* values
        will be checked against the JPEG 2000 data and warnings issued if
        different.
    version : int, optional

        * If ``1`` (default) then return the image data as an :class:`numpy.ndarray`
        * If ``2`` then return the image data as :class:`bytearray`

    Returns
    -------
    bytearray | numpy.ndarray
        The image data as either a bytearray or ndarray.

    Raises
    ------
    RuntimeError
        If the decoding failed.
    """
    buffer = BytesIO(src)
    j2k_format = _get_format(buffer)

    # Version 1
    if version == Version.v1:
        if j2k_format != 0:
            warnings.warn(
                "The (7FE0,0010) Pixel Data contains a JPEG 2000 codestream "
                "with the optional JP2 file format header, which is "
                "non-conformant to the DICOM Standard (Part 5, Annex A.4.4)"
            )

        arr = _openjpeg.decode(buffer, j2k_format, as_array=True)

        samples_per_pixel = kwargs.get("samples_per_pixel")
        bits_stored = kwargs.get("bits_stored")
        pixel_representation = kwargs.get("pixel_representation")
        no_kwargs = None in (samples_per_pixel, bits_stored, pixel_representation)

        if not ds and no_kwargs:
            return cast(np.ndarray, arr)

        ds = cast("Dataset", ds)
        samples_per_pixel = ds.get("SamplesPerPixel", samples_per_pixel)
        bits_stored = ds.get("BitsStored", bits_stored)
        pixel_representation = ds.get("PixelRepresentation", pixel_representation)

        meta = get_parameters(buffer, j2k_format)
        if samples_per_pixel != meta["nr_components"]:
            warnings.warn(
                f"The (0028,0002) Samples per Pixel value '{samples_per_pixel}' "
                f"in the dataset does not match the number of components "
                f"'{meta['nr_components']}' found in the JPEG 2000 data. "
                f"It's recommended that you change the  Samples per Pixel value "
                f"to produce the correct output"
            )

        if bits_stored != meta["precision"]:
            warnings.warn(
                f"The (0028,0101) Bits Stored value '{bits_stored}' in the "
                f"dataset does not match the component precision value "
                f"'{meta['precision']}' found in the JPEG 2000 data. "
                f"It's recommended that you change the Bits Stored value to "
                f"produce the correct output"
            )

        if bool(pixel_representation) != meta["is_signed"]:
            val = "signed" if meta["is_signed"] else "unsigned"
            ds_val = "signed" if bool(pixel_representation) else "unsigned"
            ds_val = f"'{pixel_representation}' ({ds_val})"
            warnings.warn(
                f"The (0028,0103) Pixel Representation value {ds_val} in the "
                f"dataset does not match the format of the values found in the "
                f"JPEG 2000 data '{val}'"
            )

        return cast(np.ndarray, arr)

    # Version 2
    return cast(bytearray, _openjpeg.decode(buffer, j2k_format, as_array=False))


def get_parameters(
    stream: Union[str, os.PathLike, bytes, bytearray, BinaryIO],
    j2k_format: Union[int, None] = None,
) -> Dict[str, Union[int, str, bool]]:
    """Return a :class:`dict` containing the JPEG2000 image parameters.

    .. versionchanged:: 1.1

        `stream` can now also be :class:`str` or :class:`pathlib.Path`

    Parameters
    ----------
    stream : str, pathlib.Path, bytes or file-like
        The path to the JPEG 2000 file or a Python object containing the
        encoded JPEG 2000 data. If using a file-like then the object must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    j2k_format : int, optional
        The JPEG 2000 format to use for decoding, one of:

        * ``0``: JPEG-2000 codestream (such as from DICOM *Pixel Data*)
        * ``1``: JPT-stream (JPEG 2000, JPIP)
        * ``2``: JP2 file format

    Returns
    -------
    dict
        A :class:`dict` containing the J2K image parameters:
        ``{'columns': int, 'rows': int, 'colourspace': str,
        'nr_components: int, 'precision': int, `is_signed`: bool}``. Possible
        colour spaces are "unknown", "unspecified", "sRGB", "monochrome",
        "YUV", "e-YCC" and "CYMK".

    Raises
    ------
    RuntimeError
        If reading the image parameters failed.
    """
    if isinstance(stream, (str, Path)):
        with open(stream, "rb") as f:
            buffer: BinaryIO = BytesIO(f.read())
            buffer.seek(0)
    elif isinstance(stream, (bytes, bytearray)):
        buffer = BytesIO(stream)
    else:
        # BinaryIO
        required_methods = ["read", "tell", "seek"]
        if not all([hasattr(stream, meth) for meth in required_methods]):
            raise TypeError(
                "The Python object containing the encoded JPEG 2000 data must "
                "either be bytes or have read(), tell() and seek() methods."
            )
        buffer = cast(BinaryIO, stream)

    if j2k_format is None:
        j2k_format = _get_format(buffer)

    if j2k_format not in [0, 1, 2]:
        raise ValueError(f"Unsupported 'j2k_format' value: {j2k_format}")

    return cast(
        Dict[str, Union[str, int, bool]],
        _openjpeg.get_parameters(buffer, j2k_format),
    )


def encode(
    arr,
    codec=0,
    bits_stored=8,
    photometric_interpretation=0,
    lossless=1,
    use_mct=1,
    compression_ratio=0,
):
    return _openjpeg.encode(
        arr,
        codec,
        bits_stored,
        photometric_interpretation,
        lossless,
        use_mct,
        compression_ratio
    )

"""
Supported Encoding

JPEG 2000 Lossless Only
-----------------------

+----------------+-----------+----------------+-----------+--------+
| Photometric    | Samples   | Pixel          | Bits      | Bits   |
| Interpretation | per Pixel | Representation | Allocated | Stored |
+================+===========+================+===========+========+
| PALETTE COLOR  | 1         | 0              | 8, 16     | 1-16   |
+----------------+-----------+----------------+-----------+--------+
| MONOCHROME1    | 1         | 0 or 1         | 8, 16     | 1-16   |
| MONOCHROME2    |           |                |           |        |
+----------------+-----------+----------------+-----------+--------+
| YBR_RCT        | 3         | 0              | 8, 16     | 1-16   |
| RGB            |           |                |           |        |
| YBR_FULL       |           |                |           |        |
+----------------+-----------+----------------+-----------+--------+

JPEG 2000
---------

+----------------+-----------+----------------+-----------+--------+
| Photometric    | Samples   | Pixel          | Bits      | Bits   |
| Interpretation | per Pixel | Representation | Allocated | Stored |
+================+===========+================+===========+========+
| MONOCHROME1    | 1         | 0 or 1         | 8, 16     | 1-16   |
| MONOCHROME2    |           |                |           |        |
+----------------+-----------+----------------+-----------+--------+
| YBR_ICT        | 3         | 0              | 8, 16     | 1-16   |
| RGB            |           |                |           |        |
| YBR_FULL       |           |                |           |        |
+----------------+-----------+----------------+-----------+--------+

Notes:
* Test with 24 and 32 bit (signed/unsigned integers) input
* Only allow access to the encoding parameters that will be customised by pydicom

Fixed Parameters
----------------
* 1 tile
* Precinct 2**15 x 2**15
* Code block 64 x 64
* 6 resolutions
* No SOP or EPH markers
* No subsampling
* Progression order LRCP
* No index file
* No ROI upshift
* No image origin offset
* No tile origin offset
* No JPWL

Configurable
------------
* Codec: 0 (J2C) or 2 (JP2) only
* Colourspace: 0-5
* Lossless or lossy
  * (Lossless) Reversible DWT 5-3, MCT if photometric interpretation is RGB
  * (Lossy) Irreversible DWT
  * (Lossless) Allow MCT on or off, default on with RGB input
  * (Lossy) compression ratio?

"""
# class J2KRProfile:
#     def __init__(self):
#         # opj_image_comp.data is INT32 so max 32 bits? But may not work
#         #   test to 16 bits allocated/stored, disallow above that
#         # 1 or 3 samples per pixel
#         # 8, 16, 24 or 32 bits allocated
#         # 1-32 bits stored
#         # PALETTE COLOR, MONOCHROME1, MONOCHROME2, YBR_RCT, RGB, YBR_FULL
#         # MCT 1 if RGB -> YBR_RCT
#         # Lossless
#
# class J2KIProfile:
#     def __init__(self):
#         # 1 or 3 samples per pixel
#         # 8, 16, 24, 32 or 40 bits allocated
#         # 1-38 bits stored
#         pass
#
# class EncodingOptions:
#     def __init__(self):
#         # opj_cparameters
#         # opj_cparameters_t
#
#         # opj_set_default_encoder_parameters -> then customise
#         #
#
#         self.nr_tiles = 1
#         self.subsampling = (1, 1)
#         self.progression_order = 0  # LRCP 0 | RLCP 1 | RPCL 2 | PCRL 3 | CPRL 4
#         self.block_size = (64, 64)
#         self.lossless = True
#         # UNKNOWN -1 | J2K 0 | JPT 1 | JP2 2 | JPP 3 | JPX 4
#         # J2K: JPEG2000 codestream
#         # JPT J2K + JPIP - read only?
#         # JP2
#         # JPP - not coded?
#         # JPX - not coded?
#         self.codec = 0
#           # if samples >= 3 UNKNOWN -1 | UNSPECIFIED 0 | SRGB 1 | GRAY 2 | SYCC 3 | EYCC 4 | CMYK 5
#         self.colourspace = "YCC"
#         self.nr_resolutions = 6
#         self.sop_marker = False
#         self.eph_marker = False
#         self.thing = "Reversible DWT 5-3"
#         self.compression_ratio = None
#         # no ROI upshift
#         # no origin offsets
#         # no JPWL
#
#         # require input -> ndarray | buffer-like
#         # require output -> bytes
#         # require output format -> J2C?
#         # width, height, samples, bit depth, {s, u}@<dx1>x<dy1>... (if RAW)
#
#         # Optional: compression ratio
