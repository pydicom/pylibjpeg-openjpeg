"""Tests for the pylibjpeg pixel data handler."""

import pytest

try:
    from pydicom import __version__
    from pydicom.encaps import generate_pixel_data_frame
    from pydicom.pixel_data_handlers.util import (
        reshape_pixel_array,
        pixel_dtype,
    )

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

from openjpeg import get_parameters, decode_pixel_data
from openjpeg.data import get_indexed_datasets

if HAS_PYDICOM:
    PYD_VERSION = int(__version__.split(".")[0])


def generate_frames(ds):
    """Return a frame generator for DICOM datasets."""
    nr_frames = ds.get("NumberOfFrames", 1)
    return generate_pixel_data_frame(ds.PixelData, nr_frames)


@pytest.mark.skipif(not HAS_PYDICOM, reason="pydicom unavailable")
class TestHandler:
    """Tests for the pixel data handler."""

    def test_invalid_type_raises(self):
        """Test decoding using invalid type raises."""
        index = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        ds = index["MR_small_jp2klossless.dcm"]["ds"]
        frame = tuple(next(generate_frames(ds)))
        assert not hasattr(frame, "tell") and not isinstance(frame, bytes)

        msg = "a bytes-like object is required, not 'tuple'"
        with pytest.raises(TypeError, match=msg):
            decode_pixel_data(frame)

    def test_no_dataset(self):
        index = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        ds = index["MR_small_jp2klossless.dcm"]["ds"]
        frame = next(generate_frames(ds))
        arr = decode_pixel_data(frame)
        assert arr.flags.writeable
        assert "uint8" == arr.dtype
        length = ds.Rows * ds.Columns * ds.SamplesPerPixel * ds.BitsAllocated / 8
        assert (length,) == arr.shape


class HandlerTestBase:
    """Baseclass for handler tests."""

    uid = None

    def setup_method(self):
        self.ds = get_indexed_datasets(self.uid)

    def plot(self, arr, index=None, cmap=None):
        import matplotlib.pyplot as plt

        if index is not None:
            if cmap:
                plt.imshow(arr[index], cmap=cmap)
            else:
                plt.imshow(arr[index])
        else:
            if cmap:
                plt.imshow(arr, cmap=cmap)
            else:
                plt.imshow(arr)

        plt.show()


@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestLibrary:
    """Tests for libjpeg itself."""

    def test_non_conformant_raises(self):
        """Test that a non-conformant JPEG image raises an exception."""
        ds_list = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        # Image has invalid Se value in the SOS marker segment
        item = ds_list["966.dcm"]
        assert 0xC000 == item["Status"][1]
        msg = r"Error decoding the J2K data: failed to decode image"
        with pytest.raises(RuntimeError, match=msg):
            item["ds"].pixel_array

    def test_valid_no_warning(self, recwarn):
        """Test no warning issued when dataset matches JPEG data."""
        index = get_indexed_datasets("1.2.840.10008.1.2.4.90")
        ds = index["966_fixed.dcm"]["ds"]
        ds.pixel_array

        assert len(recwarn) == 0


# ISO/IEC 10918 JPEG - Expected fail
@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEGBaseline(HandlerTestBase):
    """Test the handler with ISO 10918 JPEG images.

    1.2.840.10008.1.2.4.50 : JPEG Baseline (Process 1)
    """

    uid = "1.2.840.10008.1.2.4.50"

    def test_raises(self):
        """Test greyscale."""
        ds = self.ds["JPEGBaseline_1s_1f_u_08_08.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        if PYD_VERSION < 3:
            msg = (
                "Unable to convert the Pixel Data as the 'pylibjpeg-libjpeg' plugin is "
                "not installed"
            )
        else:
            msg = (
                r"Unable to decompress 'JPEG Baseline \(Process 1\)' pixel data because "
                "all plugins are missing dependencies:"
            )

        with pytest.raises(RuntimeError, match=msg):
            ds.pixel_array


@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEGExtended(HandlerTestBase):
    """Test the handler with ISO 10918 JPEG images.

    1.2.840.10008.1.2.4.51 : JPEG Extended (Process 2 and 4)
    """

    uid = "1.2.840.10008.1.2.4.51"

    # Process 4
    def test_raises(self):
        """Test process 4 greyscale."""
        ds = self.ds["RG2_JPLY_fixed.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        # Input precision is 12, not 10
        assert 10 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        if PYD_VERSION < 3:
            msg = (
                "Unable to convert the Pixel Data as the 'pylibjpeg-libjpeg' plugin is "
                "not installed"
            )
        else:
            msg = (
                r"Unable to decompress 'JPEG Extended \(Process 2 and 4\)' pixel data because "
                "all plugins are missing dependencies:"
            )

        with pytest.raises(RuntimeError, match=msg):
            ds.pixel_array


@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEGLossless(HandlerTestBase):
    """Test the handler with ISO 10918 JPEG images.

    1.2.840.10008.1.2.4.57 : JPEG Lossless, Non-Hierarchical (Process 14)
    """

    uid = "1.2.840.10008.1.2.4.57"

    def test_raises(self):
        """Test process 2 greyscale."""
        ds = self.ds["JPEGLossless_1s_1f_u_16_12.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 12 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        if PYD_VERSION < 3:
            msg = (
                "Unable to convert the Pixel Data as the 'pylibjpeg-libjpeg' plugin is "
                "not installed"
            )
        else:
            msg = (
                r"Unable to decompress 'JPEG Lossless, Non-Hierarchical \(Process "
                r"14\)' pixel data because all plugins are missing dependencies:"
            )

        with pytest.raises(RuntimeError, match=msg):
            ds.pixel_array


@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEGLosslessSV1(HandlerTestBase):
    """Test the handler with ISO 10918 JPEG images.

    1.2.840.10008.1.2.4.70 : JPEG Lossless, Non-Hierarchical, First-Order
    Prediction (Process 14 [Selection Value 1]
    """

    uid = "1.2.840.10008.1.2.4.70"

    def test_raises(self):
        """Test process 2 greyscale."""
        ds = self.ds["JPEGLosslessP14SV1_1s_1f_u_08_08.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        if PYD_VERSION < 3:
            msg = (
                "Unable to convert the Pixel Data as the 'pylibjpeg-libjpeg' plugin is "
                "not installed"
            )
        else:
            msg = (
                "Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order "
                r"Prediction \(Process 14 \[Selection Value 1\]\)' "
                "pixel data because all plugins are missing dependencies:"
            )

        with pytest.raises(RuntimeError, match=msg):
            ds.pixel_array


# ISO/IEC 14495 JPEG-LS - Expected fail
@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEGLSLossless(HandlerTestBase):
    """Test the handler with ISO 14495 JPEG-LS images.

    1.2.840.10008.1.2.4.80 : JPEG-LS Lossless Image Compression
    """

    uid = "1.2.840.10008.1.2.4.80"

    def test_raises(self):
        """Test process 2 greyscale."""
        ds = self.ds["MR_small_jpeg_ls_lossless.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        if PYD_VERSION < 3:
            msg = (
                "Unable to convert the Pixel Data as the 'pylibjpeg-libjpeg' plugin is "
                "not installed"
            )
        else:
            msg = (
                r"Unable to decompress 'JPEG-LS Lossless Image Compression' "
                "pixel data because all plugins are missing dependencies:"
            )

        with pytest.raises(RuntimeError, match=msg):
            ds.pixel_array


@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEGLS(HandlerTestBase):
    """Test the handler with ISO 14495 JPEG-LS images.

    1.2.840.10008.1.2.4.81 : JPEG-LS Lossy (Near-Lossless) Image Compression
    """

    uid = "1.2.840.10008.1.2.4.81"

    def test_raises(self):
        """Test process 2 greyscale."""
        ds = self.ds["CT1_JLSN.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        if PYD_VERSION < 3:
            msg = (
                "Unable to convert the Pixel Data as the 'pylibjpeg-libjpeg' plugin is "
                "not installed"
            )
        else:
            msg = (
                r"Unable to decompress 'JPEG-LS Lossy \(Near-Lossless\) Image Compression' "
                "pixel data because all plugins are missing dependencies:"
            )

        with pytest.raises(RuntimeError, match=msg):
            ds.pixel_array


# ISO/IEC 15444 JPEG 2000
@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEG2000Lossless(HandlerTestBase):
    """Test the handler with ISO 15444 JPEG2000 images.

    1.2.840.10008.1.2.4.90 : JPEG 2000 Image Compression (Lossless Only)
    """

    uid = "1.2.840.10008.1.2.4.90"

    @pytest.mark.skip("No suitable dataset")
    def test_1s_1f_i_08_08(self):
        """Test 1 component, 1 frame, signed 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "int8" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    @pytest.mark.skip("No suitable dataset")
    def test_1s_1f_u_08_08(self):
        """Test 1 component, 1 frame, unsigned 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "int8" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    @pytest.mark.skip("No suitable dataset")
    def test_1s_2f_i_08_08(self):
        """Test 1 component, 2 frame, signed 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 2 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "int8" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    def test_3s_1f_u_08_08(self):
        """Test 3 component, 1 frame, unsigned 8-bit."""
        ds = self.ds["US1_J2KR.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 3 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "YBR_RCT" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "uint8" == arr.dtype
        assert (ds.Rows, ds.Columns, ds.SamplesPerPixel) == arr.shape

        # Values checked against GDCM
        assert [
            [180, 26, 0],
            [172, 15, 0],
            [162, 9, 0],
            [152, 4, 0],
            [145, 0, 0],
            [132, 0, 0],
            [119, 0, 0],
            [106, 0, 0],
            [87, 0, 0],
            [37, 0, 0],
            [0, 0, 0],
            [50, 0, 0],
            [100, 0, 0],
            [109, 0, 0],
            [122, 0, 0],
            [135, 0, 0],
            [145, 0, 0],
            [155, 5, 0],
            [165, 11, 0],
            [175, 17, 0],
        ] == arr[175:195, 28, :].tolist()

    @pytest.mark.skip("No suitable dataset")
    def test_3s_2f_i_08_08(self):
        """Test 3 component, 2 frame, signed 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "RGB" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "uint8" == arr.dtype
        assert (ds.Rows, ds.Columns, ds.SamplesPerPixel) == arr.shape

    def test_1s_1f_i_16_14(self):
        """Test 1 component, 1 frame, signed 16/14-bit."""
        ds = self.ds["693_J2KR.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        # assert 14 == ds.BitsStored   # wrong bits stored value - should warn?
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<i2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [1022, 1051, 1165, 1442, 1835, 2096, 2074, 1868, 1685, 1603] == arr[
            290, 135:145
        ].tolist()
        assert -2000 == arr[0, 0]

    def test_1s_1f_i_16_16(self):
        """Test 1 component, 1 frame, signed 16/16-bit."""
        ds = self.ds["explicit_VR-UN.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<i2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [21, 287, 797, 863, 813, 428, 55, -7, 37, -22] == arr[
            142, 260:270
        ].tolist()

    def test_1s_1f_u_16_12(self):
        """Test 1 component, 1 frame, unsigned 16/12-bit."""
        ds = self.ds["NM_Kakadu44_SOTmarkerincons.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 12 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [2719, 2678, 2684, 2719, 2882, 2963, 2981, 2949, 3049, 3145] == arr[
            2417, 1223:1233
        ].tolist()

    def test_1s_1f_u_16_15(self):
        """Test 1 component, 1 frame, unsigned 16/15-bit."""
        ds = self.ds["RG1_J2KR.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 15 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [21920, 22082, 22245, 22406, 22557, 22619, 22629, 22724, 22787] == arr[
            45:54, 184
        ].tolist()

    def test_1s_1f_u_16_16(self):
        """Test 1 component, 1 frame, unsigned 16/16-bit."""
        ds = self.ds["JPEG2KLossless_1s_1f_u_16_16.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [55680, 57220, 58518, 61083, 64624, 65535, 65535, 65535, 65535] == arr[
            1402:1411, 1388
        ].tolist()

    def test_1s_10f_u_16_16(self):
        """Test 1 component, 10 frame, unsigned 16/16-bit."""
        ds = self.ds["emri_small_jpeg_2k_lossless.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 10 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 12 == ds.BitsStored  # wrong bits stored value
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.NumberOfFrames, ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [290, 312, 345, 414, 379, 239, 119, 76, 80, 76] == arr[
            0, 36, 41:51
        ].tolist()
        assert [87, 45, 193, 341, 307, 133, 54, 99, 113, 101] == arr[
            9, 53:63, 57
        ].tolist()

    @pytest.mark.skip("No suitable dataset")
    def test_1s_2f_i_16_16(self):
        """Test 1 component, 2 frame, signed 16/16-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

    @pytest.mark.skip("No suitable dataset")
    def test_3s_1f_i_16_16(self):
        """Test 3 component, 1 frame, signed 16/16-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

    @pytest.mark.skip("No suitable dataset")
    def test_3s_2f_i_16_16(self):
        """Test 3 component, 2 frame, signed 16/16-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

    def test_jp2(self):
        """Test decoding a non-conformant Pixel Data with JP2 data."""
        ds = self.ds["GDCMJ2K_TextGBR.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 3 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "YBR_RCT" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "uint8" == arr.dtype
        assert (ds.Rows, ds.Columns, ds.SamplesPerPixel) == arr.shape

        # Values checked against GDCM
        assert [255, 0, 0] == arr[54, 145, :].tolist()
        assert [0, 255, 0] == arr[179, 85, :].tolist()
        assert [0, 0, 255] == arr[275, 38, :].tolist()
        assert [128, 128, 128] == arr[368, 376, :].tolist()

    def test_data_unsigned_pr_1(self):
        """Test unsigned JPEG data with Pixel Representation 1"""
        ds = self.ds["TOSHIBA_J2K_SIZ0_PixRep1.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        # Note: if PR is 1 but the JPEG data is unsigned then it should
        #   probably be converted to signed using 2s complement
        ds.pixel_array
        frame = next(generate_frames(ds))
        params = get_parameters(frame)
        assert params["is_signed"] is False

        # self.plot(arr)

    def test_data_signed_pr_0(self):
        """Test signed JPEG data with Pixel Representation 0"""
        ds = self.ds["TOSHIBA_J2K_SIZ1_PixRep0.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        # Note: if PR is 0 but the JPEG data is signed then... ?
        ds.pixel_array

        frame = next(generate_frames(ds))
        params = get_parameters(frame)
        assert params["is_signed"] is True


@pytest.mark.skipif(not HAS_PYDICOM, reason="No dependencies")
class TestJPEG2000(HandlerTestBase):
    """Test the handler with ISO 15444 JPEG2000 images.

    1.2.840.10008.1.2.4.91 : JPEG 2000 Image Compression
    """

    uid = "1.2.840.10008.1.2.4.91"

    @pytest.mark.skip("No suitable dataset")
    def test_1s_1f_i_08_08(self):
        """Test 1 component, 1 frame, signed 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "int8" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    @pytest.mark.skip("No suitable dataset")
    def test_1s_1f_u_08_08(self):
        """Test 1 component, 1 frame, unsigned 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "int8" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    @pytest.mark.skip("No suitable dataset")
    def test_1s_2f_i_08_08(self):
        """Test 1 component, 2 frame, signed 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 2 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "int8" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    def test_3s_1f_u_08_08(self):
        """Test 3 component, 1 frame, unsigned 8-bit."""
        ds = self.ds["SC_rgb_gdcm_KY.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 3 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "RGB" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "uint8" == arr.dtype
        assert (ds.Rows, ds.Columns, ds.SamplesPerPixel) == arr.shape

        assert [255, 0, 0] == arr[5, 0].tolist()
        assert [255, 128, 128] == arr[15, 0].tolist()
        assert [0, 255, 0] == arr[25, 0].tolist()
        assert [128, 255, 128] == arr[35, 0].tolist()
        assert [0, 0, 255] == arr[45, 0].tolist()
        assert [128, 128, 255] == arr[55, 0].tolist()
        assert [0, 0, 0] == arr[65, 0].tolist()
        assert [64, 64, 64] == arr[75, 0].tolist()
        assert [192, 192, 192] == arr[85, 0].tolist()
        assert [255, 255, 255] == arr[95, 0].tolist()

    @pytest.mark.skip("No suitable dataset")
    def test_3s_2f_i_08_08(self):
        """Test 3 component, 2 frame, signed 8-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "RGB" in ds.PhotometricInterpretation
        assert 8 == ds.BitsAllocated
        assert 8 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "uint8" == arr.dtype
        assert (ds.Rows, ds.Columns, ds.SamplesPerPixel) == arr.shape

    @pytest.mark.skip("No suitable dataset")
    def test_1s_1f_i_16_14(self):
        """Test 1 component, 1 frame, signed 16/14-bit."""
        ds = self.ds["693_J2KR.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        # assert 14 == ds.BitsStored   # wrong bits stored value - should warn?
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<i2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

    def test_1s_1f_i_16_16(self):
        """Test 1 component, 1 frame, signed 16/16-bit."""
        ds = self.ds["693_J2KI.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 14 == ds.BitsStored  # wrong bits stored
        assert 1 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<i2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [812, 894, 1179, 1465, 1751, 2037, 1939, 1841, 1743, 1645] == arr[
            290, 135:145
        ].tolist()
        assert -2016 == arr[0, 0]

    def test_1s_1f_u_16_12(self):
        """Test 1 component, 1 frame, unsigned 16/12-bit."""
        ds = self.ds["MR2_J2KI.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 12 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [27, 31, 32, 25, 17, 7, 6, 36, 63, 39] == arr[770:780, 136].tolist()

    def test_1s_1f_u_16_15(self):
        """Test 1 component, 1 frame, unsigned 16/15-bit."""
        ds = self.ds["RG1_J2KI.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 15 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [24291, 24345, 24401, 24455, 24508, 24559, 24604, 24647, 24687] == arr[
            175:184, 28
        ].tolist()

    def test_1s_1f_u_16_16(self):
        """Test 1 component, 1 frame, unsigned 16/16-bit."""
        ds = self.ds["OsirixFake16BitsStoredFakeSpacing.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [1090, 1114, 1135, 1123, 1100, 1100, 1086, 1093, 1128, 1141] == arr[
            107, 66:76
        ].tolist()

    @pytest.mark.skip("No suitable dataset")
    def test_1s_10f_u_16_16(self):
        """Test 1 component, 10 frame, unsigned 16/16-bit."""
        ds = self.ds["emri_small_jpeg_2k_lossless.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 10 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        # assert 16 == ds.BitsStored  # wrong bits stored value - should warn?
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.NumberOfFrames, ds.Rows, ds.Columns) == arr.shape

    @pytest.mark.skip("No suitable dataset")
    def test_1s_2f_i_16_16(self):
        """Test 1 component, 2 frame, signed 16/16-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

    @pytest.mark.skip("No suitable dataset")
    def test_3s_1f_i_16_16(self):
        """Test 3 component, 1 frame, signed 16/16-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

    @pytest.mark.skip("No suitable dataset")
    def test_3s_2f_i_16_16(self):
        """Test 3 component, 2 frame, signed 16/16-bit."""
        ds = self.ds[".dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 16 == ds.BitsStored
        assert 1 == ds.PixelRepresentation

    def test_jp2(self):
        """Test decoding a non-conformant Pixel Data with JP2 data."""
        ds = self.ds["MAROTECH_CT_JP2Lossy.dcm"]["ds"]
        assert self.uid == ds.file_meta.TransferSyntaxUID
        assert 1 == ds.SamplesPerPixel
        assert 1 == getattr(ds, "NumberOfFrames", 1)
        assert "MONOCHROME" in ds.PhotometricInterpretation
        assert 16 == ds.BitsAllocated
        assert 12 == ds.BitsStored
        assert 0 == ds.PixelRepresentation

        arr = ds.pixel_array
        assert arr.flags.writeable
        assert "<u2" == arr.dtype
        assert (ds.Rows, ds.Columns) == arr.shape

        # Values checked against GDCM
        assert [939, 698, 988, 1074, 1029, 1218, 1471, 873, 647, 864] == arr[
            191, 136:146
        ].tolist()
