
from openjpeg.utils import decode_pixel_data


DICOM_DECODERS = {
     # JPEG 2000 Lossless
    '1.2.840.10008.1.2.4.90' : decode_pixel_data,
     # JPEG 2000
    '1.2.840.10008.1.2.4.91' : decode_pixel_data,
}
"""A :class:`dict` of the DICOM (0002,0010) *Transfer Syntax UID* values
that specify the encoding of the (7FE0,0010) *Pixel Data* that can be decoded
using this library. The format of the ``dict`` is  ``{UID: callable}``, where
`callable` the function to use for decoding as
``callable(arr, photometric_interpretation)``.
"""


DICOM_ENCODERS = {}
"""A :class:`dict` of the DICOM (0002,0010) *Transfer Syntax UID* values
that specify the encoding of the (7FEO,0010) *Pixel Data* to be encoded using
this library. The format of the ``dict`` is ``{UID: callable}``, where
`callable` the function to use for encoding as ``callable(arr)``.
"""
