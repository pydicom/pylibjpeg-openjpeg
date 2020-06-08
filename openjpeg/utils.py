

import _openjpeg


def get_openjpeg_version():
    """Return the openjpeg version as tuple of int."""
    version = _openjpeg.get_version().decode("ascii").split(".")
    return tuple([int(ii) for ii in version])


def decode(stream, codec_format=0, reshape=True):
    """Return the decoded JPEG2000 data from `stream` as a
    :class:`numpy.ndarray`.

    Parameters
    ----------
    stream : file-like
        A Python file-like containing the encoded JPEG 2000 data. Must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    codec_format : int, optional
        The codec format to use for decoding, one of:

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
    # Multi-component images are returned as planar configuration 1...
    #   component1 -> component2 -> component3
    arr = _openjpeg.decode(stream, codec_format)
    if not reshape:
        return arr

    meta = _libjpeg.decode(stream, codec_format)
    bpp = ceil(meta["precision"] / 8)

    dtype = f"uint{8 * bpp}"
    if meta["is_signed"]:
        dtype = f"int{8 * bpp}"

    arr = arr.view(dtype)

    shape = [meta["rows"], meta["columns"]]
    if meta["nr_components"] > 1:
        shape.append(meta["nr_components"])

    return arr.reshape(*shape)


def get_parameters(stream, codec=0):
    """Return a :class:`dict` containing the JPEG2000 image parameters.

    Parameters
    ----------
    stream : file-like
        A Python file-like containing the encoded JPEG 2000 data. Must have
        ``tell()``, ``seek()`` and ``read()`` methods.
    codec : int, optional
        The codec to use for decoding, one of:

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
    return _openjpeg.get_parameters(stream, codec)
