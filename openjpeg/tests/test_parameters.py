"""Tests for get_parameters()."""

from io import BytesIO
import os
import pytest

import numpy as np

try:
    from pydicom.encaps import generate_pixel_data_frame
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from . import add_handler, remove_handler
from openjpeg.utils import get_parameters
from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY


DIR_15444 = os.path.join(JPEG_DIRECTORY, '15444')


REF_DCM = {
    '1.2.840.10008.1.2.4.90' : [
        # filename, (rows, columns, samples/px, bits/sample, signed?)
        ('693_J2KR.dcm', (512, 512, 1, 14, True)),
        ('966_fixed.dcm', (2128, 2000, 1, 12, False)),
        ('emri_small_jpeg_2k_lossless.dcm', (64, 64, 1, 16, False)),
        ('explicit_VR-UN.dcm', (512, 512, 1, 16, True)),
        ('GDCMJ2K_TextGBR.dcm', (512, 512, 1, 16, True)),
        ('JPEG2KLossless_1s_1f_u_16_16.dcm', (1416, 1420, 1, 16, False)),
        ('MR2_J2KR.dcm', (1024, 1024, 1, 12, False)),
        ('MR_small_jp2klossless.dcm', (64, 64, 1, 16, True)),
        ('NM_Kakadu44_SOTmarkerincons.dcm', (64, 64, 1, 16, True)),
        ('RG1_J2KR.dcm', (1955, 1841, 1, 15, False)),
        ('RG3_J2KR.dcm', (1760, 1760, 1, 10, False)),
        ('TOSHIBA_J2K_OpenJPEGv2Regression.dcm', (1760, 1760, 1, 10, False)),
        ('TOSHIBA_J2K_SIZ0_PixRep1.dcm', (1760, 1760, 1, 10, False)),
        ('TOSHIBA_J2K_SIZ1_PixRep0.dcm', (1760, 1760, 1, 10, False)),
        ('US1_J2KR.dcm', (480, 640, 3, 8, False)),
    ],
    '1.2.840.10008.1.2.4.91' : [
        ('693_J2KI.dcm', (512, 512, 1, 16, True)),
        ('ELSCINT1_JP2vsJ2K.dcm', (1024, 256, 1, 16, True)),
        ('JPEG2000.dcm', (1024, 256, 1, 16, True)),
        ('MAROTECH_CT_JP2Lossy.dcm', (1024, 256, 1, 16, True)),
        ('MR2_J2KI.dcm', (1024, 1024, 1, 12, False)),
        ('OsirixFake16BitsStoredFakeSpacing.dcm', (1024, 1024, 1, 12, False)),
        ('RG1_J2KI.dcm', (1955, 1841, 1, 15, False)),
        ('RG3_J2KI.dcm', (1760, 1760, 1, 10, False)),
        ('SC_rgb_gdcm_KY.dcm', (100, 100, 3, 8, False)),
        ('US1_J2KI.dcm', (480, 640, 3, 8, False)),
    ],
}


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
class TestGetParametersDCM(object):
    """Tests for get_parameters() using DICOM datasets."""
    def generate_frames(self, ds):
        """Return a generator object with the dataset's pixel data frames."""
        nr_frames = ds.get('NumberOfFrames', 1)
        return generate_pixel_data_frame(ds.PixelData, nr_frames)

    @pytest.mark.parametrize("fname, info", REF_DCM['1.2.840.10008.1.2.4.90'])
    def test_jpeg2000r(self, fname, info):
        """Test get_parameters() for the baseline datasets."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index[fname]['ds']

        frame = next(self.generate_frames(ds))
        params = get_parameters(frame)

        assert (info[0], info[1]) == (params['rows'], params['columns'])
        assert info[2] == params['nr_components']
        assert info[3] == params['precision']
        assert info[4] == params['is_signed']

    @pytest.mark.parametrize("fname, info", REF_DCM['1.2.840.10008.1.2.4.91'])
    def test_jpeg2000i(self, fname, info):
        """Test get_parameters() for the baseline datasets."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.91')
        ds = index[fname]['ds']

        frame = next(self.generate_frames(ds))
        params = get_parameters(frame)

        assert (info[0], info[1]) == (params['rows'], params['columns'])
        assert info[2] == params['nr_components']
        assert info[3] == params['precision']
        assert info[4] == params['is_signed']

    def test_invalid_type_raises(self):
        """Test that exception is raised if invalid type for data."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['MR_small_jp2klossless.dcm']['ds']

        frame = next(self.generate_frames(ds))
        assert isinstance(frame, bytes)
        #frame =
