"""Unit tests for openjpeg."""

from io import BytesIO
import os

try:
    import pydicom
    from pydicom.encaps import generate_pixel_data_frame
    from pydicom.pixel_data_handlers.util import (
        reshape_pixel_array, get_expected_length, pixel_dtype
    )
    #from . import handler
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


import matplotlib.pyplot as plt
import numpy as np
import pytest

import openjpeg
from openjpeg.utils import get_openjpeg_version, decode

from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY


DIR_15444 = os.path.join(JPEG_DIRECTORY, '15444')

REF_DCM = {
    '1.2.840.10008.1.2.4.90' : [
        ('693_J2KR.dcm', (512, 512, 1, 16)),
        ('966_fixed.dcm', (2128, 2000, 1, 16)),
        ('emri_small_jpeg_2k_lossless.dcm', (64, 64, 1, 16)),
        ('explicit_VR-UN.dcm', (512, 512, 1, 16)),  # orientation?
        ('JPEG2KLossless_1s_1f_u_16_16.dcm', (1416, 1420, 1, 16)),  # blank?
        ('MR2_J2KR.dcm', (1024, 1024, 1, 16)),
        ('MR_small_jp2klossless.dcm', (64, 64, 1, 16)),
        ('RG1_J2KR.dcm', (1955, 1841, 1, 16)),
        ('RG3_J2KR.dcm', (1760, 1760, 1, 16)),
        ('US1_J2KR.dcm', (480, 640, 3, 8)),
    ],
    '1.2.840.10008.1.2.4.91' : [
        # filename, (rows, columns, samples/px, bits/sample)
        ('693_J2KI.dcm', (512, 512, 1, 16)),
        ('JPEG2000.dcm', (1024, 256, 1, 16)),
        ('MR2_J2KI.dcm', (1024, 1024, 1, 16)),
        ('RG1_J2KI.dcm', (1955, 1841, 1, 16)),
        ('RG3_J2KI.dcm', (1760, 1760, 1, 16)),
        ('SC_rgb_gdcm_KY.dcm', (100, 100, 3, 8)),
        ('US1_J2KI.dcm', (480, 640, 3, 8)),
    ],
}


def test_version():
    """Test that the openjpeg version can be retrieved."""
    version = get_openjpeg_version()
    assert isinstance(version, tuple)
    assert isinstance(version[0], int)
    assert 3 == len(version)
    assert 2 == version[0]


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
class TestDecodeDCM(object):
    """Tests for get_parameters() using DICOM datasets."""
    def generate_frames(self, ds):
        """Return a generator object with the dataset's pixel data frames."""
        nr_frames = ds.get('NumberOfFrames', 1)
        return generate_pixel_data_frame(ds.PixelData, nr_frames)

    @pytest.mark.parametrize("fname, info", REF_DCM['1.2.840.10008.1.2.4.90'])
    def test_jpeg2000r(self, fname, info):
        """Test get_parameters() for the j2k lossless datasets."""
        #info: (rows, columns, spp, bps)
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index[fname]['ds']

        frame = next(self.generate_frames(ds))
        length = get_expected_length(ds)
        nr_frames = getattr(ds, 'NumberOfFrames', 1)
        length = length // nr_frames
        result, arr = decode(BytesIO(frame), length)

        ds.NumberOfFrames = 1
        arr = arr.view(pixel_dtype(ds))
        if ds.SamplesPerPixel == 1:
            arr = reshape_pixel_array(ds, arr)
        else:
            arr = arr.reshape(ds.SamplesPerPixel, ds.Rows, ds.Columns)
            arr = arr.transpose(1, 2, 0)

        plt.imshow(arr)
        plt.show()


        #if info[2] == 1:
        #    assert (info[0], info[1]) == arr.shape
        #else:
        #    assert (info[0], info[1], info[2]) == arr.shape

        #if 1 <= info[3] <= 8:
        #    assert arr.dtype == 'uint8'
        #if 9 <= info[3] <= 16:
        #    assert arr.dtype == 'uint16'

    @pytest.mark.parametrize("fname, info", REF_DCM['1.2.840.10008.1.2.4.91'])
    def test_jpeg2000i(self, fname, info):
        """Test get_parameters() for the j2k datasets."""
        #info: (rows, columns, spp, bps)
        index = get_indexed_datasets('1.2.840.10008.1.2.4.91')
        ds = index[fname]['ds']

        frame = next(self.generate_frames(ds))
        result, arr = decode(BytesIO(frame), get_expected_length(ds))

        arr = arr.view(pixel_dtype(ds))
        if ds.SamplesPerPixel == 1:
            arr = reshape_pixel_array(ds, arr)
        else:
            arr = arr.reshape(ds.SamplesPerPixel, ds.Rows, ds.Columns)
            arr = arr.transpose(1, 2, 0)

        plt.imshow(arr)
        plt.show()


        #if info[2] == 1:
        #    assert (info[0], info[1]) == arr.shape
        #else:
        #    assert (info[0], info[1], info[2]) == arr.shape

        #if 1 <= info[3] <= 8:
        #    assert arr.dtype == 'uint8'
        #if 9 <= info[3] <= 16:
        #    assert arr.dtype == 'uint16'
