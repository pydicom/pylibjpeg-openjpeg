

import _openjpeg


def decode(arr, colourspace='YBR_FULL', reshape=True):
    """Return the decoded JPEG data from `arr` as a :class:`numpy.ndarray`.

    Parameters
    ----------
    arr : numpy.ndarray or bytes
        A 1D array of ``np.uint8``, or a Python :class:`bytes` object
        containing the encoded JPEG image.
    colourspace : str, optional
        One of ``'MONOCHROME1'``, ``'MONOCHROME2'``, ``'RGB'``, ``'YBR_FULL'``,
        ``'YBR_FULL_422'``.
    reshape : bool, optional
        Reshape and review the output array so it matches the image data
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
    pass
