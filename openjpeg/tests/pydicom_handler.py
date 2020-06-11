"""Use the `pylibjpeg <https://github.com/pydicom/pylibjpeg/>`_ package
to convert supported pixel data to a :class:`numpy.ndarray`.

**Supported transfer syntaxes**

* 1.2.840.10008.1.2.4.90 : JPEG2000 Lossless
* 1.2.840.10008.1.2.4.91 : JPEG2000
"""

import numpy as np
from pydicom.encaps import generate_pixel_data_frame
from pydicom.pixel_data_handlers.util import pixel_dtype, get_expected_length
from pydicom.uid import JPEG2000Lossless, JPEG2000

from openjpeg import decode_pixel_data


HANDLER_NAME = 'openjpeg-test'

DEPENDENCIES = {
    'numpy': ('http://www.numpy.org/', 'NumPy'),
}

SUPPORTED_TRANSFER_SYNTAXES = [JPEG2000Lossless, JPEG2000]


def is_available():
    """Return ``True`` if the handler has its dependencies met."""
    return True


def supports_transfer_syntax(tsyntax):
    """Return ``True`` if the handler supports the `tsyntax`.

    Parameters
    ----------
    tsyntax : uid.UID
        The Transfer Syntax UID of the *Pixel Data* that is to be used with
        the handler.
    """
    return tsyntax in SUPPORTED_TRANSFER_SYNTAXES


def needs_to_convert_to_RGB(ds):
    """Return ``True`` if the *Pixel Data* should to be converted from YCbCr to
    RGB.

    This affects JPEG transfer syntaxes.
    """
    return False


def should_change_PhotometricInterpretation_to_RGB(ds):
    """Return ``True`` if the *Photometric Interpretation* should be changed
    to RGB.

    This affects JPEG transfer syntaxes.
    """
    return False


def get_pixeldata(ds):
    """Return a :class:`numpy.ndarray` of the pixel data.

    Parameters
    ----------
    ds : Dataset
        The :class:`Dataset` containing an Image Pixel, Floating Point Image
        Pixel or Double Floating Point Image Pixel module and the
        *Pixel Data*, *Float Pixel Data* or *Double Float Pixel Data* to be
        converted. If (0028,0004) *Photometric Interpretation* is
        `'YBR_FULL_422'` then the pixel data will be
        resampled to 3 channel data as per Part 3, :dcm:`Annex C.7.6.3.1.2
        <part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2>` of the DICOM Standard.

    Returns
    -------
    np.ndarray
        The contents of (7FE0,0010) *Pixel Data* as a 1D array.
    """
    tsyntax = ds.file_meta.TransferSyntaxUID
    # The check of transfer syntax must be first
    if tsyntax not in SUPPORTED_TRANSFER_SYNTAXES:
        raise NotImplementedError(
            "Unable to convert the pixel data as the transfer syntax "
            "is not supported by the openjpeg pixel data handler."
        )

    # Check required elements
    required_elements = [
        'BitsAllocated', 'Rows', 'Columns', 'PixelRepresentation',
        'SamplesPerPixel', 'PhotometricInterpretation', 'PixelData',
    ]
    missing = [elem for elem in required_elements if elem not in ds]
    if missing:
        raise AttributeError(
            "Unable to convert the pixel data as the following required "
            "elements are missing from the dataset: " + ", ".join(missing)
        )

    # Calculate the expected length of the pixel data (in bytes)
    #   Note: this does NOT include the trailing null byte for odd length data
    expected_len = get_expected_length(ds)

    # How long each frame is in bytes
    nr_frames = getattr(ds, 'NumberOfFrames', 1)
    frame_len = expected_len // nr_frames

    # The decoded data will be placed here
    arr = np.empty(expected_len, np.uint8)

    # Generators for the encoded JPG image frame(s) and insertion offsets
    generate_frames = generate_pixel_data_frame(ds.PixelData, nr_frames)
    generate_offsets = range(0, expected_len, frame_len)
    for frame, offset in zip(generate_frames, generate_offsets):
        # Encoded JPG data to be sent to the decoder
        arr[offset:offset + frame_len] = decode_pixel_data(
            frame, ds.group_dataset(0x0028)
        )

    return arr.view(pixel_dtype(ds))
