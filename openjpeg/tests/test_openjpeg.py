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
from openjpeg.utils import get_openjpeg_version, decode, get_parameters

from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY


DIR_15444 = os.path.join(JPEG_DIRECTORY, '15444')

REF_DCM = {
    '1.2.840.10008.1.2.4.90' : [
        # filename, (rows, columns, samples/px, bits/sample, signed?)
        ('693_J2KR.dcm', (512, 512, 1, 14, True)),
        ('966_fixed.dcm', (2128, 2000, 1, 12, False)),
        ('emri_small_jpeg_2k_lossless.dcm', (64, 64, 1, 16, False)),
        ('explicit_VR-UN.dcm', (512, 512, 1, 16, True)),
        ('GDCMJ2K_TextGBR.dcm', (400, 400, 3, 8, False)),
        ('JPEG2KLossless_1s_1f_u_16_16.dcm', (1416, 1420, 1, 16, False)),
        ('MR_small_jp2klossless.dcm', (64, 64, 1, 16, True)),
        ('MR2_J2KR.dcm', (1024, 1024, 1, 12, False)),
        ('NM_Kakadu44_SOTmarkerincons.dcm', (2500, 2048, 1, 12, False)),
        ('RG1_J2KR.dcm', (1955, 1841, 1, 15, False)),
        ('RG3_J2KR.dcm', (1760, 1760, 1, 10, False)),
        ('TOSHIBA_J2K_OpenJPEGv2Regression.dcm', (512, 512, 1, 16, True)),
        ('TOSHIBA_J2K_SIZ0_PixRep1.dcm', (512, 512, 1, 16, True)),
        ('TOSHIBA_J2K_SIZ1_PixRep0.dcm', (512, 512, 1, 16, False)),
        ('US1_J2KR.dcm', (480, 640, 3, 8, False)),
    ],
    '1.2.840.10008.1.2.4.91' : [
        ('693_J2KI.dcm', (512, 512, 1, 16, True)),
        ('ELSCINT1_JP2vsJ2K.dcm', (512, 512, 1, 12, False)),
        ('JPEG2000.dcm', (1024, 256, 1, 16, True)),
        ('MAROTECH_CT_JP2Lossy.dcm', (716, 512, 1, 12, False)),
        ('MR2_J2KI.dcm', (1024, 1024, 1, 12, False)),
        ('OsirixFake16BitsStoredFakeSpacing.dcm', (224, 176, 1, 16, False)),
        ('RG1_J2KI.dcm', (1955, 1841, 1, 15, False)),
        ('RG3_J2KI.dcm', (1760, 1760, 1, 10, False)),
        ('SC_rgb_gdcm_KY.dcm', (100, 100, 3, 8, False)),
        ('US1_J2KI.dcm', (480, 640, 3, 8, False)),
    ],
}


def test_version():
    """Test that the openjpeg version can be retrieved."""
    version = get_openjpeg_version()
    assert isinstance(version, tuple)
    assert isinstance(version[0], int)
    assert 3 == len(version)
    assert 2 == version[0]


def generate_frames(ds):
    nr_frames = ds.get('NumberOfFrames', 1)
    return generate_pixel_data_frame(ds.PixelData, nr_frames)


def test_get_parameters():
    index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
    ds = index['MR2_J2KR.dcm']['ds']
    frame = next(generate_frames(ds))
    result = get_parameters(BytesIO(frame))
    print(result)

    ds = index['US1_J2KR.dcm']['ds']
    frame = next(generate_frames(ds))
    result = get_parameters(BytesIO(frame))
    print(result)

    index = get_indexed_datasets('1.2.840.10008.1.2.4.91')
    ds = index['SC_rgb_gdcm_KY.dcm']['ds']
    frame = next(generate_frames(ds))
    result = get_parameters(BytesIO(frame))
    print(result)

    ds = index['US1_J2KI.dcm']['ds']
    frame = next(generate_frames(ds))
    result = get_parameters(BytesIO(frame))
    print(result)


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
        arr = decode(BytesIO(frame), reshape=False)

        ds.NumberOfFrames = 1
        arr = arr.view(pixel_dtype(ds))
        arr = reshape_pixel_array(ds, arr)

        plt.imshow(arr)
        plt.show()

        if info[2] == 1:
            assert (info[0], info[1]) == arr.shape
        else:
            assert (info[0], info[1], info[2]) == arr.shape

        if 1 <= info[3] <= 8:
            if info[4] == 1:
                assert arr.dtype == 'int8'
            else:
                assert arr.dtype == 'uint8'
        if 9 <= info[3] <= 16:
            if info[4] == 1:
                assert arr.dtype == 'int16'
            else:
                assert arr.dtype == 'uint16'

    @pytest.mark.parametrize("fname, info", REF_DCM['1.2.840.10008.1.2.4.91'])
    def test_jpeg2000i(self, fname, info):
        """Test get_parameters() for the j2k datasets."""
        #info: (rows, columns, spp, bps)
        index = get_indexed_datasets('1.2.840.10008.1.2.4.91')
        ds = index[fname]['ds']

        frame = next(self.generate_frames(ds))
        arr = decode(BytesIO(frame), reshape=False)

        ds.NumberOfFrames = 1
        arr = arr.view(pixel_dtype(ds))
        arr = reshape_pixel_array(ds, arr)

        plt.imshow(arr)
        plt.show()

        if info[2] == 1:
            assert (info[0], info[1]) == arr.shape
        else:
            assert (info[0], info[1], info[2]) == arr.shape

        if 1 <= info[3] <= 8:
            if info[4] == 1:
                assert arr.dtype == 'int8'
            else:
                assert arr.dtype == 'uint8'
        if 9 <= info[3] <= 16:
            if info[4] == 1:
                assert arr.dtype == 'int16'
            else:
                assert arr.dtype == 'uint16'
