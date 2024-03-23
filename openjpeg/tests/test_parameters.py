"""Tests for get_parameters()."""

import pytest

try:
    from pydicom.encaps import generate_pixel_data_frame

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from openjpeg import get_parameters
from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY


DIR_15444 = JPEG_DIRECTORY / "15444"


REF_DCM = {
    "1.2.840.10008.1.2.4.90": [
        # filename, (rows, columns, samples/px, bits/sample, signed?)
        ("693_J2KR.dcm", (512, 512, 1, 14, True)),
        ("966_fixed.dcm", (2128, 2000, 1, 12, False)),
        ("emri_small_jpeg_2k_lossless.dcm", (64, 64, 1, 16, False)),
        ("explicit_VR-UN.dcm", (512, 512, 1, 16, True)),
        ("GDCMJ2K_TextGBR.dcm", (400, 400, 3, 8, False)),
        ("JPEG2KLossless_1s_1f_u_16_16.dcm", (1416, 1420, 1, 16, False)),
        ("MR_small_jp2klossless.dcm", (64, 64, 1, 16, True)),
        ("MR2_J2KR.dcm", (1024, 1024, 1, 12, False)),
        ("NM_Kakadu44_SOTmarkerincons.dcm", (2500, 2048, 1, 16, False)),
        ("RG1_J2KR.dcm", (1955, 1841, 1, 15, False)),
        ("RG3_J2KR.dcm", (1760, 1760, 1, 10, False)),
        ("TOSHIBA_J2K_OpenJPEGv2Regression.dcm", (512, 512, 1, 16, False)),
        ("TOSHIBA_J2K_SIZ0_PixRep1.dcm", (512, 512, 1, 16, False)),
        ("TOSHIBA_J2K_SIZ1_PixRep0.dcm", (512, 512, 1, 16, True)),
        ("US1_J2KR.dcm", (480, 640, 3, 8, False)),
    ],
    "1.2.840.10008.1.2.4.91": [
        ("693_J2KI.dcm", (512, 512, 1, 16, True)),
        ("ELSCINT1_JP2vsJ2K.dcm", (512, 512, 1, 12, False)),
        ("JPEG2000.dcm", (1024, 256, 1, 16, True)),
        ("MAROTECH_CT_JP2Lossy.dcm", (716, 512, 1, 12, False)),
        ("MR2_J2KI.dcm", (1024, 1024, 1, 12, False)),
        ("OsirixFake16BitsStoredFakeSpacing.dcm", (224, 176, 1, 11, False)),
        ("RG1_J2KI.dcm", (1955, 1841, 1, 15, False)),
        ("RG3_J2KI.dcm", (1760, 1760, 1, 10, False)),
        ("SC_rgb_gdcm_KY.dcm", (100, 100, 3, 8, False)),
        ("US1_J2KI.dcm", (480, 640, 3, 8, False)),
    ],
}


def generate_frames(ds):
    """Return a frame generator for DICOM datasets."""
    nr_frames = ds.get("NumberOfFrames", 1)
    return generate_pixel_data_frame(ds.PixelData, nr_frames)


def test_bad_decode():
    """Test trying to decode bad data."""
    stream = b"\xff\x4f\xff\x51\x00\x00\x01"
    msg = r"Error decoding the J2K data: failed to read the header"
    with pytest.raises(RuntimeError, match=msg):
        get_parameters(stream)


def test_subsampling():
    """Test parameters with subsampled data (see #36)."""
    jpg = DIR_15444 / "2KLS" / "oj36.j2k"
    params = get_parameters(jpg)
    assert params["rows"] == 256
    assert params["columns"] == 256
    assert params["colourspace"] == "unspecified"
    assert params["samples_per_pixel"] == 3
    assert params["precision"] == 8
    assert params["is_signed"] is False
    assert params["nr_tiles"] == 0


@pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
class TestGetParametersDCM:
    """Tests for get_parameters() using DICOM datasets."""

    @pytest.mark.parametrize("fname, info", REF_DCM["1.2.840.10008.1.2.4.90"])
    def test_jpeg2000r(self, fname, info):
        """Test get_parameters() for the baseline datasets."""
        index = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        ds = index[fname]["ds"]

        frame = next(generate_frames(ds))
        params = get_parameters(frame)

        assert (info[0], info[1]) == (params["rows"], params["columns"])
        assert info[2] == params["samples_per_pixel"]
        assert info[3] == params["precision"]
        assert info[4] == params["is_signed"]

    @pytest.mark.parametrize("fname, info", REF_DCM["1.2.840.10008.1.2.4.91"])
    def test_jpeg2000i(self, fname, info):
        """Test get_parameters() for the baseline datasets."""
        index = get_indexed_datasets("1.2.840.10008.1.2.4.91")
        ds = index[fname]["ds"]

        frame = next(generate_frames(ds))
        params = get_parameters(frame)

        assert (info[0], info[1]) == (params["rows"], params["columns"])
        assert info[2] == params["samples_per_pixel"]
        assert info[3] == params["precision"]
        assert info[4] == params["is_signed"]

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_bad_type_raises(self):
        """Test decoding using invalid type raises."""
        index = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        ds = index["MR_small_jp2klossless.dcm"]["ds"]
        frame = tuple(next(generate_frames(ds)))
        assert not hasattr(frame, "tell") and not isinstance(frame, bytes)

        msg = (
            r"The Python object containing the encoded JPEG 2000 data must "
            r"either be bytes or have read\(\), tell\(\) and seek\(\) methods."
        )
        with pytest.raises(TypeError, match=msg):
            get_parameters(frame)

    @pytest.mark.skipif(not HAS_PYDICOM, reason="No pydicom")
    def test_decode_format_raises(self):
        """Test decoding using invalid format raises."""
        index = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        ds = index["693_J2KR.dcm"]["ds"]
        frame = next(generate_frames(ds))
        msg = r"Unsupported 'j2k_format' value: 3"
        with pytest.raises(ValueError, match=msg):
            get_parameters(frame, j2k_format=3)
