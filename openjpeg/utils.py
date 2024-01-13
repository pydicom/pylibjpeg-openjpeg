from enum import IntEnum
from io import BytesIO
import logging
from math import ceil, log
import os
from pathlib import Path
from typing import BinaryIO, Tuple, Union, TYPE_CHECKING, Any, Dict, cast, List
import warnings

import numpy as np

import _openjpeg


if TYPE_CHECKING:  # pragma: no cover
    from pydicom.dataset import Dataset


LOGGER = logging.getLogger(__name__)


class Version(IntEnum):
    v1 = 1
    v2 = 2


class PhotometricInterpretation(IntEnum):
    MONOCHROME1 = 2
    MONOCHROME2 = 2
    PALETTE_COLOR = 2
    RGB = 1
    YBR_FULL = 3
    YBR_FULL_422 = 3


MAGIC_NUMBERS = {
    # JPEG 2000 codestream, has no header, .j2k, .jpc, .j2c
    b"\xff\x4f\xff\x51": 0,
    # JP2 and JP2 RFC3745, .jp2
    b"\x0d\x0a\x87\x0a": 2,
    b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a": 2,
    # JPT, .jpt - shrug
}
DECODING_ERRORS = {
    1: "failed to create the input stream",
    2: "failed to setup the decoder",
    3: "failed to read the header",
    4: "failed to set the component indices",
    5: "failed to set the decoded area",
    6: "failed to decode image",
    7: "support for more than 16-bits per component is not implemented",
    8: "failed to upscale subsampled components",
}
ENCODING_ERRORS = {
    # Validation errors
    1: (
        "the input array has an unsupported number of samples per pixel, "
        "must be 1, 3 or 4"
    ),
    2: (
        "the input array has an invalid shape, must be (rows, columns) or "
        "(rows, columns, planes)"
    ),
    3: ("the input array has an unsupported number of rows, must be " "in (1, 65535)"),
    4: (
        "the input array has an unsupported number of columns, must be " "in (1, 65535)"
    ),
    5: (
        "the input array has an unsupported dtype, only bool, u1, u2, u4, i1, i2"
        " and i4 are supported"
    ),
    6: "the input array must use little endian byte ordering",
    7: "the input array must be C-style, contiguous and aligned",
    8: (
        "the image precision given by bits stored must be in (1, itemsize of "
        "the input array's dtype)"
    ),
    9: (
        "the value of the 'photometric_interpretation' parameter is not valid "
        "for the number of samples per pixel in the input array"
    ),
    10: "the valid of the 'codec_format' paramter is invalid",
    11: "more than 100 'compression_ratios' is not supported",
    12: "invalid item in the 'compression_ratios' value",
    13: "invalid compression ratio, must be in the range (1, 1000)",
    14: "more than 100 'signal_noise_ratios' is not supported",
    15: "invalid item in the 'signal_noise_ratios' value",
    16: "invalid signal-to-noise ratio, must be in the range (0, 1000)",
    # Encoding errors
    20: "failed to assign the image component parameters",
    21: "failed to create an empty image object",
    22: "failed to set the encoding handler",
    23: "failed to set up the encoder",
    24: "failed to create the output stream",
    25: "failure result from 'opj_start_compress()'",
    26: "failure result from 'opj_encode()'",
    27: "failure result from 'opj_endt_compress()'",
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

    return_code, arr = _openjpeg.decode(buffer, j2k_format, as_array=True)
    if return_code != 0:
        raise RuntimeError(
            f"Error decoding the J2K data: {DECODING_ERRORS.get(return_code, return_code)}"
        )

    if not reshape:
        return cast(np.ndarray, arr)

    meta = get_parameters(buffer, j2k_format)
    precision = cast(int, meta["precision"])
    rows = cast(int, meta["rows"])
    columns = cast(int, meta["columns"])
    pixels_per_sample = cast(int, meta["nr_components"])
    pixel_representation = cast(bool, meta["is_signed"])
    bpp = ceil(precision / 8)

    bpp = 4 if bpp == 3 else bpp
    dtype = f"u{bpp}" if not pixel_representation else f"i{bpp}"
    arr = arr.view(dtype)

    shape = [rows, columns]
    if pixels_per_sample > 1:
        shape.append(pixels_per_sample)

    return cast(np.ndarray, arr.reshape(*shape))


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

        return_code, arr = _openjpeg.decode(buffer, j2k_format, as_array=True)
        if return_code != 0:
            raise RuntimeError(
                f"Error decoding the J2K data: {DECODING_ERRORS.get(return_code, return_code)}"
            )

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
    return_code, buffer = _openjpeg.decode(buffer, j2k_format, as_array=False)
    if return_code != 0:
        raise RuntimeError(
            f"Error decoding the J2K data: {DECODING_ERRORS.get(return_code, return_code)}"
        )

    return cast(bytearray, buffer)


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


def _get_bits_stored(arr: np.ndarray) -> int:
    """Return a 'bits_stored' appropriate for `arr`."""
    if arr.dtype.kind == "b":
        return 1

    maximum = arr.max()
    if arr.dtype.kind == "u":
        if maximum == 0:
            return 1

        return int(log(maximum, 2) + 1)

    minimum = arr.min()
    for bit_depth in range(1, arr.dtype.itemsize * 8):
        if maximum <= 2 ** (bit_depth - 1) - 1 and minimum >= -(2 ** (bit_depth - 1)):
            return bit_depth

    return cast(int, arr.dtype.itemsize * 8)


def encode(
    arr: np.ndarray,
    bits_stored: Union[int, None] = None,
    photometric_interpretation: int = 0,
    use_mct: bool = True,
    compression_ratios: Union[List[float], None] = None,
    signal_noise_ratios: Union[List[float], None] = None,
    codec_format: int = 0,
    **kwargs: Any,
) -> bytes:
    """Return the JPEG 2000 compressed `arr`.

    Encoding of the input array will use lossless compression by default, to
    use lossy compression either `compression_ratios` or `signal_noise_ratios`
    must be supplied.

    Parameters
    ----------
    arr : numpy.ndarray
        The array containing the image data to be encoded. 1-bit data should be
        unpacked (if packed) and stored as a bool or u1 dtype.
    bits_stored : int, optional
        The bit-depth of the image, defaults to the minimum bit-depth required
        to fully cover the range of pixel data in `arr`.
    photometric_interpretation : int, optional
        The colour space of the unencoded image data that will be set in the
        JPEG 2000 metadata. If you are encoded RGB or YCbCr data then it's
        strongly recommended that you select the correct colour space so
        that anyone decoding your image data will know which colour space
        transforms to apply. One of:

        * ``0``: Unspecified
        * ``1``: sRGB
        * ``2``: Greyscale
        * ``3``: sYCC (YCbCr)
        * ``4``: eYCC
        * ``5``: CMYK
    use_mct : bool, optional
        Apply multi-component transformation (MCT) prior to encoding the image
        data. Defaults to ``True`` when the `photometric_interpretation` is
        ``1`` as it is intended for use with RGB data and should result in
        smaller file sizes. For all other values of `photometric_interpretation`
        the value of `use_mct` will be ignored and MCT not applied.

        If MCT is applied then the corresponding value of (0028,0004)
        *Photometric Interpretation* is:

        * ``"YBR_RCT"`` for lossless encoding
        * ``"YBR_ICT"`` for lossy encoding

        If MCT is not applied then *Photometric Intrepretation* should be the
        value corresponding to the unencoded dataset.
    compression_ratios : list[float], optional
        Required if using lossy encoding, this is the compression ratio to use
        for each layer. Should be in decreasing order (such as ``[80, 30, 10]``)
        and the final value may be ``1`` to indicate lossless encoding should
        be used for that layer. Cannot be used with `signal_noise_ratios`.
    signal_noise_ratios : list[float], optional
        Required if using lossy encoding, this is the desired peak
        signal-to-noise ratio to use for each layer. Should be in increasing
        order (such as ``[30, 40, 50]``), except for the last layer which may
        be ``0`` to indicate lossless encoding should be used for that layer.
        Cannot be used with `compression_ratios`.
    codec_format : int, optional
        The codec to used when encoding:

        * ``0``: JPEG 2000 codestream only (default) (J2K/J2C format)
        * ``2``: A boxed JPEG 2000 codestream (JP2 format)

    Returns
    -------
    bytes
        The encoded JPEG 2000 image data.
    """
    if compression_ratios is None:
        compression_ratios = []

    if signal_noise_ratios is None:
        signal_noise_ratios = []

    if arr.dtype.kind not in ("b", "i", "u"):
        raise ValueError(
            f"The input array has an unsupported dtype '{arr.dtype}', only "
            "bool, u1, u2, u4, i1, i2 and i4 are supported"
        )

    if bits_stored is None:
        bits_stored = _get_bits_stored(arr)

    # The destination for the encoded data, must support BinaryIO
    return_code, buffer = _openjpeg.encode(
        arr,
        bits_stored,
        photometric_interpretation,
        use_mct,
        compression_ratios,
        signal_noise_ratios,
        codec_format,
    )

    if return_code != 0:
        raise RuntimeError(
            f"Error encoding the data: {ENCODING_ERRORS.get(return_code, return_code)}"
        )

    return cast(bytes, buffer)


def encode_pixel_data(arr: np.ndarray, **kwargs: Any) -> bytes:
    """Return the JPEG 2000 compressed `arr`.

    .. versionadded:: 2.1

    Parameters
    ----------
    src : numpy.ndarray
        An array containing a single frame of image data to be encoded.
    **kwargs
        The following keyword arguments are optional:

        * ``'photometric_interpretation'``: int - the colour space of the
          unencoded image in `arr`, one of the following:

          * ``0``: Unspecified
          * ``1``: sRGB
          * ``2``: Greyscale
          * ``3``: sYCC (YCbCr)
          * ``4``: eYCC
          * ``5``: CMYK

          It is strongly recommended you supply an appropriate
          `photometric_interpretation`.
        * ``'bits_stored'``: int - the bit-depth of the pixels in the image,
          if not supplied then the smallest bit-depth that covers the full range
          of pixel data in `arr` will be used.
        * ``'lossless'``: bool: ``True`` to use lossless encoding (default),
          ``False`` for lossy encoding. If using lossy encoding then
          `compression_ratios` is required.
        * ```use_mct': bool: ``True`` to use MCT with RGB images (default),
          ``False`` otherwise.
        * ``'codec_format'``: int - 0 for a JPEG2000 codestream (default),
          1 for a JP2 codestream
        * ''`compression_ratios'``: list[float] - required if `lossless` is
          ``False``. Should be a list of the desired compression ratio per layer
          in decreasing values. The final layer may be 1 for lossless
          compression of that layer. Minimum value is 1 (for lossless).

    Returns
    -------
    bytes
        The JPEG 2000 encoded image data.
    """
    return encode(arr, **kwargs)
