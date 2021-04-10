"""Unit tests for openjpeg."""

from io import BytesIO
import os

try:
    import pydicom
    from pydicom.encaps import generate_pixel_data_frame
    from pydicom.pixel_data_handlers.util import (
        reshape_pixel_array, get_expected_length, pixel_dtype
    )
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

import numpy as np
import pytest

from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY
from openjpeg.utils import get_openjpeg_version, decode, get_parameters


DIR_15444 = JPEG_DIRECTORY / '15444'


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
        ('NM_Kakadu44_SOTmarkerincons.dcm', (2500, 2048, 1, 16, False)),
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
    """Return a frame generator for DICOM datasets."""
    nr_frames = ds.get('NumberOfFrames', 1)
    return generate_pixel_data_frame(ds.PixelData, nr_frames)


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
def test_bad_decode():
    """Test trying to decode bad data."""
    index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
    ds = index['966.dcm']['ds']
    frame = next(generate_frames(ds))
    msg = r"Error decoding the J2K data: failed to decode image"
    with pytest.raises(RuntimeError, match=msg):
        decode(frame)


class TestDecode(object):
    """General tests for decode."""
    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_bytes(self):
        """Test decoding using bytes."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['MR_small_jp2klossless.dcm']['ds']
        frame = next(generate_frames(ds))
        assert isinstance(frame, bytes)
        arr = decode(frame)
        assert arr.flags.writeable
        assert 'int16' == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # It'd be nice to standardise the pixel value testing...
        assert (422, 319, 361) == tuple(arr[0, 31:34])
        assert (366, 363, 322) == tuple(arr[31, :3])
        assert (1369, 1129, 862) == tuple(arr[-1, -3:])
        assert 862 == arr[-1, -1]

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_filelike(self):
        """Test decoding using file-like."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['MR_small_jp2klossless.dcm']['ds']
        frame = BytesIO(next(generate_frames(ds)))
        assert isinstance(frame, BytesIO)
        arr = decode(frame)
        assert arr.flags.writeable
        assert 'int16' == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # It'd be nice to standardise the pixel value testing...
        assert (422, 319, 361) == tuple(arr[0, 31:34])
        assert (366, 363, 322) == tuple(arr[31, :3])
        assert (1369, 1129, 862) == tuple(arr[-1, -3:])
        assert 862 == arr[-1, -1]

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_bad_type_raises(self):
        """Test decoding using invalid type raises."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['MR_small_jp2klossless.dcm']['ds']
        frame = tuple(next(generate_frames(ds)))
        assert not hasattr(frame, 'tell') and not isinstance(frame, bytes)

        msg = (
            r"The Python object containing the encoded JPEG 2000 data must "
            r"either be bytes or have read\(\), tell\(\) and seek\(\) methods."
        )
        with pytest.raises(TypeError, match=msg):
            decode(frame)

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_bad_format_raises(self):
        """Test decoding using invalid jpeg format raises."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['MR_small_jp2klossless.dcm']['ds']
        frame = next(generate_frames(ds))

        msg = r"Unsupported 'j2k_format' value: 3"
        with pytest.raises(ValueError, match=msg):
            decode(frame, j2k_format=3)

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_reshape_true(self):
        """Test decoding using invalid jpeg format raises."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['US1_J2KR.dcm']['ds']
        frame = next(generate_frames(ds))

        arr = decode(frame)
        assert arr.flags.writeable
        assert (ds.Rows, ds.Columns, ds.SamplesPerPixel) == arr.shape

        # Values checked against GDCM
        assert [
            [180,  26,   0],
            [172,  15,   0],
            [162,   9,   0],
            [152,   4,   0],
            [145,   0,   0],
            [132,   0,   0],
            [119,   0,   0],
            [106,   0,   0],
            [ 87,   0,   0],
            [ 37,   0,   0],
            [  0,   0,   0],
            [ 50,   0,   0],
            [100,   0,   0],
            [109,   0,   0],
            [122,   0,   0],
            [135,   0,   0],
            [145,   0,   0],
            [155,   5,   0],
            [165,  11,   0],
            [175,  17,   0]] == arr[175:195, 28, :].tolist()

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_reshape_false(self):
        """Test decoding using invalid jpeg format raises."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['US1_J2KR.dcm']['ds']
        frame = next(generate_frames(ds))

        arr = decode(frame, reshape=False)
        assert arr.flags.writeable
        assert (ds.Rows * ds.Columns * ds.SamplesPerPixel,) == arr.shape

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_signed_error(self):
        """Regression test for #30."""
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index['693_J2KR.dcm']['ds']
        frame = next(generate_frames(ds))

        arr = decode(frame)
        assert -2000 == arr[0, 0]

    def test_decode_subsampled(self):
        """Test decoding subsampled data (see #36)."""
        # Component 1 is (1, 1)
        # Component 2 is (2, 1)
        # Component 3 is (2, 1)
        jpg = DIR_15444 / "2KLS" / "oj36.j2k"
        with open(jpg, 'rb') as f:
            arr = decode(f.read())

        assert arr.flags.writeable
        assert 'uint8' == arr.dtype
        assert (256, 256, 3) == arr.shape
        assert [235, 244, 245] == arr[0, 0, :].tolist()


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
class TestDecodeDCM(object):
    """Tests for get_parameters() using DICOM datasets."""
    @pytest.mark.parametrize("fname, info", REF_DCM['1.2.840.10008.1.2.4.90'])
    def test_jpeg2000r(self, fname, info):
        """Test get_parameters() for the j2k lossless datasets."""
        #info: (rows, columns, spp, bps)
        index = get_indexed_datasets('1.2.840.10008.1.2.4.90')
        ds = index[fname]['ds']
        frame = next(generate_frames(ds))
        arr = decode(BytesIO(frame), reshape=False)
        assert arr.flags.writeable

        ds.NumberOfFrames = 1
        arr = arr.view(pixel_dtype(ds))
        arr = reshape_pixel_array(ds, arr)

        #plt.imshow(arr)
        #plt.show()

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

        frame = next(generate_frames(ds))
        arr = decode(BytesIO(frame), reshape=False)
        assert arr.flags.writeable

        ds.NumberOfFrames = 1
        arr = arr.view(pixel_dtype(ds))
        arr = reshape_pixel_array(ds, arr)

        #plt.imshow(arr)
        #plt.show()

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
