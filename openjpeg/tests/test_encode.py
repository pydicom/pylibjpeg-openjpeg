import math
from struct import unpack
import sys

import numpy as np
import pytest

from openjpeg.data import JPEG_DIRECTORY
from openjpeg.utils import (
    encode_array,
    encode_buffer,
    encode_pixel_data,
    decode,
    PhotometricInterpretation as PI,
    _get_bits_stored,
)


DIR_15444 = JPEG_DIRECTORY / "15444"


def parse_j2k(buffer):
    # SOC -> SIZ -> COD -> (COC) -> QCD -> (QCC) -> (RGN)
    # soc = buffer[:2]  # SOC box, 0xff 0x4f

    # Should be at the start of the SIZ marker
    # siz = buffer[2:4]  # 0xff 0x51
    # l_siz = buffer[4:6]  # length of SIZ box
    # r_siz = buffer[6:8]
    # x_siz = buffer[8:12]
    # y_siz = buffer[12:16]
    # xo_siz = buffer[16:20]
    # yo_siz = buffer[20:24]
    # xt_siz = buffer[24:28]
    # yt_siz = buffer[28:32]
    # xto_siz = buffer[32:36]
    # yto_siz = buffer[36:40]
    c_siz = buffer[40:42]
    nr_components = unpack(">H", c_siz)[0]

    o = 42
    for component in range(nr_components):
        ssiz = buffer[o]
        # xrsiz = buffer[o + 1]
        # yrsiz = buffer[o + 2]
        o += 3

    # Should be at the start of the COD marker
    # cod = buffer[o : o + 2]
    # l_cod = buffer[o + 2 : o + 4]
    # s_cod = buffer[o + 4 : o + 5]
    sg_cod = buffer[o + 5 : o + 9]

    # progression_order = sg_cod[0]
    nr_layers = sg_cod[1:3]
    mct = sg_cod[3]  # 0 for none, 1 for applied

    param = {}
    if ssiz & 0x80:
        param["precision"] = (ssiz & 0x7F) + 1
        param["is_signed"] = True
    else:
        param["precision"] = ssiz + 1
        param["is_signed"] = False

    param["components"] = nr_components
    param["mct"] = bool(mct)
    param["layers"] = unpack(">H", nr_layers)[0]

    return param


class TestEncode:
    """Tests for encode_array()"""

    def test_invalid_shape_raises(self):
        """Test invalid array shapes raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an invalid shape, "
            r"must be \(rows, columns\) or \(rows, columns, planes\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1,), dtype="u1"))

        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2, 3, 4), dtype="u1"))

    def test_invalid_samples_per_pixel_raises(self):
        """Test invalid samples per pixel raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an unsupported number "
            "of samples per pixel, must be 1, 3 or 4"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2, 2), dtype="u1"))

        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2, 5), dtype="u1"))

    def test_invalid_dtype_raises(self):
        """Test invalid array dtype raise exceptions."""
        msg = "input array has an unsupported dtype"
        for dtype in ("u8", "i8", "f", "d", "c", "U", "m", "M"):
            with pytest.raises((ValueError, RuntimeError), match=msg):
                encode_array(np.ones((1, 2), dtype=dtype))

    def test_invalid_contiguity_raises(self):
        """Test invalid array contiguity raise exceptions."""
        msg = (
            "Error encoding the data: the input array must be C-style, "
            "contiguous and aligned"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((3, 3), dtype="u1").T)

    def test_invalid_dimensions_raises(self):
        """Test invalid array dimensions raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an unsupported number "
            r"of rows, must be in \[1, 65535\]"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((65536, 1), dtype="u1"))

        msg = (
            "Error encoding the data: the input array has an unsupported number "
            r"of columns, must be in \[1, 65535\]"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 65536), dtype="u1"))

    def test_invalid_bits_stored_raises(self):
        """Test invalid bits_stored"""
        msg = (
            "Invalid value for the 'bits_stored' parameter, the value must "
            r"be in the range \(1, 8\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), bits_stored=0)

        with pytest.raises(ValueError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), bits_stored=9)

        msg = (
            "A 'bits_stored' value of 15 is incompatible with the range of "
            r"pixel data in the input array: \(-16385, 2\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode_array(np.asarray([-(2**14) - 1, 2], dtype="<i2"), bits_stored=15)

        msg = (
            "A 'bits_stored' value of 15 is incompatible with the range of "
            r"pixel data in the input array: \(-16384, 16384\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode_array(np.asarray([-(2**14), 2**14], dtype="<i2"), bits_stored=15)

        msg = (
            "A 'bits_stored' value of 4 is incompatible with the range of "
            r"pixel data in the input array: \(0, 16\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode_array(np.asarray([0, 2**4], dtype="<u2"), bits_stored=4)

    def test_invalid_pixel_value_raises(self):
        """Test invalid pixel values raise exceptions."""
        msg = (
            "The input array contains values outside the range of the maximum "
            "supported bit-depth of 24"
        )
        with pytest.raises(ValueError, match=msg):
            encode_array(np.asarray([2**24, 2], dtype="<u4"))

        with pytest.raises(ValueError, match=msg):
            encode_array(np.asarray([2**23, 2], dtype="<i4"))

        with pytest.raises(ValueError, match=msg):
            encode_array(np.asarray([-(2**23) - 1, 2], dtype="<i4"))

    def test_compression_snr_raises(self):
        """Test using compression_ratios and signal_noise_ratios raises."""
        msg = (
            "Only one of 'compression_ratios' or 'signal_noise_ratios' is "
            "allowed when performing lossy compression"
        )
        with pytest.raises(ValueError, match=msg):
            encode_array(
                np.asarray([0, 2], dtype="<u2"),
                compression_ratios=[2, 1],
                signal_noise_ratios=[1, 2],
            )

    def test_invalid_photometric_raises(self):
        """Test invalid photometric_interpretation raises."""
        msg = (
            "Error encoding the data: the value of the 'photometric_interpretation' "
            "parameter is not valid for the number of samples per pixel"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), photometric_interpretation=PI.RGB)

        with pytest.raises(RuntimeError, match=msg):
            encode_array(
                np.ones((1, 2, 3), dtype="u1"),
                photometric_interpretation=PI.MONOCHROME2,
            )

        with pytest.raises(RuntimeError, match=msg):
            encode_array(
                np.ones((1, 2, 4), dtype="u1"),
                photometric_interpretation=PI.MONOCHROME2,
            )

        with pytest.raises(RuntimeError, match=msg):
            encode_array(
                np.ones((1, 2, 4), dtype="u1"), photometric_interpretation=PI.RGB
            )

    def test_invalid_compression_ratios_raises(self):
        """Test an invalid 'compression_ratios' raises exceptions."""
        msg = "More than 10 compression layers is not supported"
        with pytest.raises(ValueError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), compression_ratios=[1] * 11)

        msg = (
            "Error encoding the data: invalid compression ratio, lowest value "
            "must be at least 1"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), compression_ratios=[0])

    def test_invalid_signal_noise_ratios_raises(self):
        """Test an invalid 'signal_noise_ratios' raises exceptions."""
        msg = "More than 10 compression layers is not supported"
        with pytest.raises(ValueError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), signal_noise_ratios=[1] * 11)

        msg = (
            "Error encoding the data: invalid signal-to-noise ratio, lowest "
            "value must be at least 0"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"), signal_noise_ratios=[-1])

    def test_encoding_failures_raise(self):
        """Miscellaneous test to check that failures are handled properly."""
        # Not exhaustive! Missing coverage for
        #   assigning image components
        #   creating the empty image
        #   setting the encoder
        #   creating the output stream
        #   opj_encode_array()
        #   opj_end_compress()

        # Input too small
        msg = r"Error encoding the data: failure result from 'opj_start_compress\(\)'"
        with pytest.raises(RuntimeError, match=msg):
            encode_array(np.ones((1, 2), dtype="u1"))

    def test_mct(self):
        """Test that MCT is applied as required."""
        # Should only be applied with RGB
        arr = np.random.randint(0, 2**8 - 1, size=(100, 100, 3), dtype="u1")
        buffer = encode_array(arr, photometric_interpretation=PI.RGB, use_mct=True)
        param = parse_j2k(buffer)
        assert param["mct"] is True

        buffer = encode_array(
            arr,
            photometric_interpretation=PI.RGB,
            use_mct=True,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is True

        buffer = encode_array(arr, photometric_interpretation=PI.RGB, use_mct=False)
        param = parse_j2k(buffer)
        assert param["mct"] is False

        buffer = encode_array(
            arr,
            photometric_interpretation=PI.RGB,
            use_mct=False,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        for pi in (0, 3, 4):
            buffer = encode_array(arr, photometric_interpretation=pi, use_mct=True)
            param = parse_j2k(buffer)
            assert param["mct"] is False

            buffer = encode_array(
                arr,
                photometric_interpretation=pi,
                use_mct=True,
                compression_ratios=[2.5, 3, 5],
            )
            param = parse_j2k(buffer)
            assert param["mct"] is False

        arr = np.random.randint(0, 2**8 - 1, size=(100, 100), dtype="u1")
        buffer = encode_array(
            arr, photometric_interpretation=PI.MONOCHROME1, use_mct=True
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        buffer = encode_array(
            arr,
            photometric_interpretation=PI.MONOCHROME1,
            use_mct=True,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        arr = np.random.randint(0, 2**8 - 1, size=(100, 100, 4), dtype="u1")
        buffer = encode_array(arr, photometric_interpretation=5, use_mct=True)
        param = parse_j2k(buffer)
        assert param["mct"] is False

        buffer = encode_array(
            arr,
            photometric_interpretation=5,
            use_mct=True,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

    def test_lossless_bool(self):
        """Test encoding bool data for bit-depth 1"""
        """Test encoding bool data for bit-depth 1"""
        # Convert one of the test images to 1-bit
        # Note that as of OpenJpeg v2.5.0 that random 1-bit images are prone
        #   to encoding failures
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)
        assert "uint8" == arr.dtype
        assert (480, 640, 3) == arr.shape

        mono = arr[..., 0]
        mono[mono <= 127] = 0
        mono[mono > 127] = 1
        mono = mono.astype("bool")
        assert mono.min() == 0
        assert mono.max() == 1

        buffer = encode_array(mono, photometric_interpretation=PI.MONOCHROME2)
        out = decode(buffer)
        param = parse_j2k(buffer)
        assert param["precision"] == 1
        assert param["is_signed"] is False
        assert param["layers"] == 1
        assert param["components"] == 1

        assert out.dtype.kind == "u"
        assert np.array_equal(mono, out)

        arr[arr <= 127] = 0
        arr[arr > 127] = 1
        arr = arr.astype("bool")

        buffer = encode_array(arr, photometric_interpretation=PI.RGB)
        out = decode(buffer)
        param = parse_j2k(buffer)
        assert param["precision"] == 1
        assert param["is_signed"] is False
        assert param["layers"] == 1
        assert param["components"] == 3

        assert out.dtype.kind == "u"
        assert np.array_equal(arr, out)

    def test_lossless_unsigned(self):
        """Test encoding unsigned data for bit-depth 1-16"""
        rows = 123
        cols = 234
        for bit_depth in range(2, 17):
            maximum = 2**bit_depth - 1
            dtype = f"u{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype=dtype)
            buffer = encode_array(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, 3), dtype=dtype
            )
            buffer = encode_array(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, 4), dtype=dtype
            )
            buffer = encode_array(arr, photometric_interpretation=5, use_mct=False)
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 4

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_unsigned_u4(self):
        """Test encoding unsigned data for bit-depth 17-32"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(17, 25):
            maximum = 2**bit_depth - 1
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype="u4")
            buffer = encode_array(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, planes), dtype="u4"
            )
            buffer = encode_array(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_signed(self):
        """Test encoding signed data for bit-depth 1-16"""
        rows = 123
        cols = 543
        for bit_depth in range(2, 17):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            dtype = f"i{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype=dtype
            )
            buffer = encode_array(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            dtype = f"i{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(
                low=minimum, high=maximum, size=(rows, cols, 3), dtype=dtype
            )
            buffer = encode_array(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                low=minimum, high=maximum, size=(rows, cols, 4), dtype=dtype
            )
            buffer = encode_array(arr, photometric_interpretation=5, use_mct=False)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 4

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_lossless_signed_i4(self):
        """Test encoding signed data for bit-depth 17-32"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(17, 25):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype="i4"
            )
            buffer = encode_array(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols, planes), dtype="i4"
            )
            buffer = encode_array(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_lossy_unsigned(self):
        """Test lossy encoding with unsigned data"""
        rows = 123
        cols = 234
        for bit_depth in range(1, 17):
            maximum = 2**bit_depth - 1
            dtype = f"u{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype=dtype)
            buffer = encode_array(arr, compression_ratios=[4, 2, 1])
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.allclose(arr, out, atol=5)

            buffer = encode_array(arr, signal_noise_ratios=[50, 100, 200])
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.allclose(arr, out, atol=5)

    def test_lossy_signed(self):
        """Test lossy encoding with unsigned data"""
        rows = 123
        cols = 234
        for bit_depth in range(1, 17):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            dtype = f"i{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype=dtype
            )
            buffer = encode_array(arr, compression_ratios=[4, 2, 1])
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.allclose(arr, out, atol=5)

            buffer = encode_array(arr, signal_noise_ratios=[50, 100, 200])
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.allclose(arr, out, atol=5)

    def test_roundtrip_u1_ybr(self):
        """Test a round trip for u1 YBR."""
        jpg = DIR_15444 / "2KLS" / "oj36.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)
        assert "uint8" == arr.dtype
        assert (256, 256, 3) == arr.shape

        # Test lossless
        result = encode_array(arr, photometric_interpretation=PI.YBR_FULL)
        out = decode(result)
        assert np.array_equal(arr, out)

        # Test lossy
        result = encode_array(
            arr,
            photometric_interpretation=PI.YBR_FULL,
            compression_ratios=[6, 4, 2, 1],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

        # Test lossy
        result = encode_array(
            arr,
            photometric_interpretation=PI.YBR_FULL,
            compression_ratios=[80, 100, 150],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

    def test_roundtrip_i2_mono(self):
        """Test a round trip for i2 YBR."""

        jpg = DIR_15444 / "2KLS" / "693.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)
        assert "<i2" == arr.dtype
        assert (512, 512) == arr.shape

        if sys.byteorder == "big":
            # ndarray must match system byte order
            arr = arr.astype(">i2")

        # Test lossless
        result = encode_array(
            arr,
            photometric_interpretation=PI.MONOCHROME2,
        )
        out = decode(result)
        assert np.array_equal(arr, out)

        # Test lossy w/ compression ratios
        result = encode_array(
            arr,
            photometric_interpretation=PI.MONOCHROME2,
            compression_ratios=[6, 4, 2, 1],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

        # Test lossy w/ signal-to-noise ratios
        result = encode_array(
            arr,
            photometric_interpretation=PI.MONOCHROME2,
            signal_noise_ratios=[80, 100],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

    def test_jp2(self):
        """Test using JP2 format"""
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            arr = decode(f.read())

        buffer = encode_array(arr, codec_format=1)
        assert buffer.startswith(b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a")


class TestEncodeBuffer:
    """Tests for _openjpeg.encode_buffer"""

    def test_invalid_type_raises(self):
        """Test invalid buffer type raises."""
        msg = "'src' must be bytes or bytearray, not list"
        with pytest.raises(TypeError, match=msg):
            encode_buffer([1, 2, 3], 1, 1, 1, 1, False)

    def test_invalid_shape_raises(self):
        """Test invalid image shape raises."""
        msg = r"Invalid 'columns' value '0', must be in the range \[1, 65535\]"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 0, 1, 1, 1, False)

        msg = r"Invalid 'columns' value '65536', must be in the range \[1, 65535\]"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 65536, 1, 1, 1, False)

        msg = r"Invalid 'rows' value '0', must be in the range \[1, 65535\]"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 1, 0, 1, 1, False)

        msg = r"Invalid 'rows' value '65536', must be in the range \[1, 65535\]"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 1, 65536, 1, 1, False)

        msg = "Invalid 'samples_per_pixel' value '0', must be 1, 3 or 4"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 1, 1, 0, 1, False)

        msg = "Invalid 'samples_per_pixel' value '2', must be 1, 3 or 4"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 1, 1, 2, 1, False)

    def test_invalid_bits_stored_raises(self):
        """Test invalid image pixel precision."""
        msg = r"Invalid 'bits_stored' value '0', must be in the range \[1, 24\]"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 1, 1, 1, 0, False)

        msg = r"Invalid 'bits_stored' value '25', must be in the range \[1, 24\]"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"", 1, 1, 1, 25, False)

    def test_invalid_length_raises(self):
        """Test mismatch between actual and expected src length"""
        msg = (
            "The length of 'src' is 2 bytes which doesn't match the expected "
            "length of 1 bytes"
        )
        for bits_stored in range(1, 9):
            with pytest.raises(ValueError, match=msg):
                encode_buffer(b"\x00\x01", 1, 1, 1, bits_stored, False)

        msg = (
            "The length of 'src' is 3 bytes which doesn't match the expected "
            "length of 2 bytes"
        )
        for bits_stored in range(9, 17):
            with pytest.raises(ValueError, match=msg):
                encode_buffer(b"\x00\x01\x02", 1, 1, 1, bits_stored, False)

        msg = (
            "The length of 'src' is 3 bytes which doesn't match the expected "
            "length of 4 bytes"
        )
        for bits_stored in range(17, 25):
            with pytest.raises(ValueError, match=msg):
                encode_buffer(b"\x00\x01\x02", 1, 1, 1, bits_stored, False)

    def test_invalid_photometric_interpretation_raises(self):
        """Test invalid photometric interpretation"""
        msg = (
            "Invalid 'photometric_interpretation' value '6', must be in the "
            r"range \[0, 5\]"
        )
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"\x00", 1, 1, 1, 8, False, photometric_interpretation=6)

    def test_compression_snr_raises(self):
        """Test using compression_ratios and signal_noise_ratios raises."""
        msg = (
            "Only one of 'compression_ratios' or 'signal_noise_ratios' is "
            "allowed when performing lossy compression"
        )
        with pytest.raises(ValueError, match=msg):
            encode_buffer(
                b"\x00",
                1,
                1,
                1,
                8,
                False,
                compression_ratios=[2, 1],
                signal_noise_ratios=[1, 2],
            )

    def test_invalid_compression_ratios_raises(self):
        """Test an invalid 'compression_ratios' raises exceptions."""
        msg = "More than 10 compression layers is not supported"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"\x00", 1, 1, 1, 8, False, compression_ratios=[1] * 11)

        msg = (
            "Error encoding the data: invalid compression ratio, lowest value "
            "must be at least 1"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_buffer(b"\x00", 1, 1, 1, 8, False, compression_ratios=[0])

    def test_invalid_signal_noise_ratios_raises(self):
        """Test an invalid 'signal_noise_ratios' raises exceptions."""
        msg = "More than 10 compression layers is not supported"
        with pytest.raises(ValueError, match=msg):
            encode_buffer(b"\x00", 1, 1, 1, 8, False, signal_noise_ratios=[1] * 11)

        msg = (
            "Error encoding the data: invalid signal-to-noise ratio, lowest "
            "value must be at least 0"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_buffer(b"\x00", 1, 1, 1, 8, False, signal_noise_ratios=[-1])

    def test_invalid_pi_for_samples_raises(self):
        """Test invalid photometric interpretation values for nr of samples."""
        msg = (
            "Error encoding the data: the value of the 'photometric_interpretation' "
            "parameter is not valid for the number of samples per pixel"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode_buffer(b"\x00\x01", 1, 2, 1, 8, False, photometric_interpretation=1)

        with pytest.raises(RuntimeError, match=msg):
            encode_buffer(
                b"\x00\x01\x02", 1, 1, 3, 8, False, photometric_interpretation=2
            )

        with pytest.raises(RuntimeError, match=msg):
            encode_buffer(
                b"\x00\x01\x02\x03", 1, 1, 4, 8, False, photometric_interpretation=2
            )

    def test_encoding_failures_raise(self):
        """Miscellaneous test to check that failures are handled properly."""
        # Not exhaustive!
        # Input too small
        msg = r"Error encoding the data: failure result from 'opj_start_compress\(\)'"
        with pytest.raises(RuntimeError, match=msg):
            encode_buffer(b"\x00\x01", 1, 2, 1, 8, False)

    def test_mct(self):
        """Test that MCT is applied as required."""
        # Should only be applied with RGB
        arr = np.random.randint(0, 2**8 - 1, size=(100, 100, 3), dtype="u1")
        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            3,
            8,
            False,
            photometric_interpretation=PI.RGB,
            use_mct=True,
        )
        param = parse_j2k(buffer)
        assert param["mct"] is True

        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            3,
            8,
            False,
            photometric_interpretation=PI.RGB,
            use_mct=True,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is True

        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            3,
            8,
            False,
            photometric_interpretation=PI.RGB,
            use_mct=False,
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            3,
            8,
            False,
            photometric_interpretation=PI.RGB,
            use_mct=False,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        for pi in (0, 3, 4):
            buffer = encode_buffer(
                arr.tobytes(),
                100,
                100,
                3,
                8,
                False,
                photometric_interpretation=pi,
                use_mct=True,
            )
            param = parse_j2k(buffer)
            assert param["mct"] is False

            buffer = encode_buffer(
                arr.tobytes(),
                100,
                100,
                3,
                8,
                False,
                photometric_interpretation=pi,
                use_mct=True,
                compression_ratios=[2.5, 3, 5],
            )
            param = parse_j2k(buffer)
            assert param["mct"] is False

        arr = np.random.randint(0, 2**8 - 1, size=(100, 100), dtype="u1")
        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            1,
            8,
            False,
            photometric_interpretation=PI.MONOCHROME1,
            use_mct=True,
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            1,
            8,
            False,
            photometric_interpretation=PI.MONOCHROME1,
            use_mct=True,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        arr = np.random.randint(0, 2**8 - 1, size=(100, 100, 4), dtype="u1")
        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            4,
            8,
            False,
            photometric_interpretation=5,
            use_mct=True,
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

        buffer = encode_buffer(
            arr.tobytes(),
            100,
            100,
            4,
            8,
            False,
            photometric_interpretation=5,
            use_mct=True,
            compression_ratios=[2.5, 3, 5],
        )
        param = parse_j2k(buffer)
        assert param["mct"] is False

    def test_lossless_bool(self):
        """Test encoding bool data for bit-depth 1"""
        # Convert one of the test images to 1-bit
        # Note that as of OpenJpeg v2.5.0 that random 1-bit images are prone
        #   to encoding failures
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)
        assert "uint8" == arr.dtype
        assert (480, 640, 3) == arr.shape

        mono = arr[..., 0]
        mono[mono <= 127] = 0
        mono[mono > 127] = 1
        mono = mono.astype("bool")
        assert mono.min() == 0
        assert mono.max() == 1

        buffer = encode_buffer(
            mono.tobytes(),
            640,
            480,
            1,
            1,
            False,
            photometric_interpretation=PI.MONOCHROME2,
        )
        out = decode(buffer)
        param = parse_j2k(buffer)
        assert param["precision"] == 1
        assert param["is_signed"] is False
        assert param["layers"] == 1
        assert param["components"] == 1

        assert out.dtype.kind == "u"
        assert np.array_equal(mono, out)

        arr[arr <= 127] = 0
        arr[arr > 127] = 1
        arr = arr.astype("bool")

        buffer = encode_buffer(
            arr.tobytes(),
            640,
            480,
            3,
            1,
            False,
            photometric_interpretation=PI.RGB,
        )
        out = decode(buffer)
        param = parse_j2k(buffer)
        assert param["precision"] == 1
        assert param["is_signed"] is False
        assert param["layers"] == 1
        assert param["components"] == 3

        assert out.dtype.kind == "u"
        assert np.array_equal(arr, out)

    def test_lossless_unsigned_u1(self):
        """Test encoding unsigned data for bit-depth 1-8"""
        rows = 123
        cols = 234
        for bit_depth in range(2, 9):
            maximum = 2**bit_depth - 1
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype="u1")
            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                False,
                photometric_interpretation=PI.MONOCHROME2,
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, 3), dtype="u1"
            )
            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                3,
                bit_depth,
                False,
                photometric_interpretation=PI.RGB,
                use_mct=False,
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, 4), dtype="u1"
            )
            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                4,
                bit_depth,
                False,
                photometric_interpretation=5,
                use_mct=False,
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 4

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_unsigned_u2(self):
        """Test encoding unsigned data for bit-depth 9-16"""
        jpg = DIR_15444 / "2KLS" / "693.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)

        rows = 123
        cols = 234
        for bit_depth in range(9, 17):
            maximum = 2**bit_depth - 1
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype="u2")
            if sys.byteorder == "big":
                # I think randint() requires dtype use machine byte order
                arr = arr.byteswap().view("<u2")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                False,
                photometric_interpretation=PI.MONOCHROME2,
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, 3), dtype="u2"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<u2")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                3,
                bit_depth,
                False,
                photometric_interpretation=PI.RGB,
                use_mct=False,
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, 4), dtype="u2"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<u2")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                4,
                bit_depth,
                False,
                photometric_interpretation=5,
                use_mct=False,
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 4

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_unsigned_u4(self):
        """Test encoding unsigned data for bit-depth 17-24"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(17, 25):
            maximum = 2**bit_depth - 1
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype="u4")
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<u4")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                False,
                photometric_interpretation=PI.MONOCHROME2,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                0, high=maximum + 1, size=(rows, cols, planes), dtype="u4"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<u4")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                3,
                bit_depth,
                False,
                photometric_interpretation=PI.RGB,
                use_mct=False,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_signed_i1(self):
        """Test encoding signed data for bit-depth 1-8"""
        rows = 123
        cols = 543
        for bit_depth in range(2, 9):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype="i1"
            )
            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                True,
                photometric_interpretation=PI.MONOCHROME2,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols, 3), dtype="i1"
            )
            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                3,
                bit_depth,
                True,
                photometric_interpretation=PI.RGB,
                use_mct=False,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols, 4), dtype="i1"
            )
            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                4,
                bit_depth,
                True,
                photometric_interpretation=5,
                use_mct=False,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 4

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_lossless_signed_i2(self):
        """Test encoding signed data for bit-depth 9-16"""
        rows = 123
        cols = 543
        for bit_depth in range(9, 17):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype="i2"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<i2")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                True,
                photometric_interpretation=PI.MONOCHROME2,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols, 3), dtype="i2"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<i2")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                3,
                bit_depth,
                True,
                photometric_interpretation=PI.RGB,
                use_mct=False,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols, 4), dtype="i2"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<i2")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                4,
                bit_depth,
                True,
                photometric_interpretation=5,
                use_mct=False,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 4

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_lossless_signed_i4(self):
        """Test encoding signed data for bit-depth 17-24"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(17, 25):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype="i4"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<i4")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                True,
                photometric_interpretation=PI.MONOCHROME2,
            )
            out = decode(buffer)


            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols, planes), dtype="i4"
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view("<i4")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                3,
                bit_depth,
                True,
                photometric_interpretation=PI.RGB,
                use_mct=False,
            )
            out = decode(buffer)

            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 1
            assert param["components"] == 3

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_lossy_unsigned(self):
        """Test lossy encoding with unsigned data"""
        rows = 123
        cols = 234
        for bit_depth in range(2, 17):
            maximum = 2**bit_depth - 1
            dtype = f"u{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(0, high=maximum + 1, size=(rows, cols), dtype=dtype)
            if sys.byteorder == "big":
                arr = arr.byteswap().view(f"<{dtype}")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                False,
                compression_ratios=[4, 2, 1],
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.allclose(arr, out, atol=5)

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                False,
                signal_noise_ratios=[50, 100, 200],
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is False
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "u"
            assert np.allclose(arr, out, atol=5)

    def test_lossy_signed(self):
        """Test lossy encoding with unsigned data"""
        rows = 123
        cols = 234
        for bit_depth in range(2, 17):
            maximum = 2 ** (bit_depth - 1) - 1
            minimum = -(2 ** (bit_depth - 1))
            dtype = f"i{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(
                low=minimum, high=maximum + 1, size=(rows, cols), dtype=dtype
            )
            if sys.byteorder == "big":
                arr = arr.byteswap().view(f"<{dtype}")

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                True,
                compression_ratios=[4, 2, 1],
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.allclose(arr, out, atol=5)

            buffer = encode_buffer(
                arr.tobytes(),
                cols,
                rows,
                1,
                bit_depth,
                True,
                signal_noise_ratios=[50, 100, 200],
            )
            out = decode(buffer)
            param = parse_j2k(buffer)
            assert param["precision"] == bit_depth
            assert param["is_signed"] is True
            assert param["layers"] == 3
            assert param["components"] == 1

            assert out.dtype.kind == "i"
            assert np.allclose(arr, out, atol=5)

    def test_roundtrip_u1_ybr(self):
        """Test a round trip for u1 YBR."""
        jpg = DIR_15444 / "2KLS" / "oj36.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)
        assert "uint8" == arr.dtype
        assert (256, 256, 3) == arr.shape

        # Test lossless
        result = encode_buffer(
            arr.tobytes(),
            256,
            256,
            3,
            8,
            False,
            photometric_interpretation=PI.YBR_FULL,
        )
        out = decode(result)
        assert np.array_equal(arr, out)

        # Test lossy
        result = encode_buffer(
            arr.tobytes(),
            256,
            256,
            3,
            8,
            False,
            photometric_interpretation=PI.YBR_FULL,
            compression_ratios=[6, 4, 2, 1],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

        # Test lossy
        result = encode_buffer(
            arr.tobytes(),
            256,
            256,
            3,
            8,
            False,
            photometric_interpretation=PI.YBR_FULL,
            compression_ratios=[80, 100, 150],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

    def test_roundtrip_i2_mono(self):
        """Test a round trip for i2 YBR."""

        jpg = DIR_15444 / "2KLS" / "693.j2k"
        with open(jpg, "rb") as f:
            data = f.read()

        arr = decode(data)
        assert "<i2" == arr.dtype
        assert (512, 512) == arr.shape

        # Test lossless
        result = encode_buffer(
            arr.tobytes(),
            512,
            512,
            1,
            14,
            True,
            photometric_interpretation=PI.MONOCHROME2,
        )
        out = decode(result)
        assert np.array_equal(arr, out)

        # Test lossy w/ compression ratios
        result = encode_buffer(
            arr.tobytes(),
            512,
            512,
            1,
            14,
            True,
            photometric_interpretation=PI.MONOCHROME2,
            compression_ratios=[6, 4, 2, 1],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

        # Test lossy w/ signal-to-noise ratios
        result = encode_buffer(
            arr.tobytes(),
            512,
            512,
            1,
            14,
            True,
            photometric_interpretation=PI.MONOCHROME2,
            signal_noise_ratios=[80, 100],
        )
        out = decode(result)
        assert np.allclose(out, arr, atol=2)

    def test_jp2(self):
        """Test using JP2 format"""
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            arr = decode(f.read())

        kwargs = {
            "rows": 480,
            "columns": 640,
            "samples_per_pixel": 3,
            "pixel_representation": 0,
            "is_signed": False,
            "bits_stored": 8,
            "codec_format": 1,
        }

        buffer = encode_buffer(arr.tobytes(), **kwargs)
        assert buffer.startswith(b"\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a")


class TestEncodePixelData:
    """Tests for encode_pixel_data()"""

    def test_nominal(self):
        """Test the function works OK"""
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            arr = decode(f.read())

        kwargs = {
            "rows": 480,
            "columns": 640,
            "samples_per_pixel": 3,
            "pixel_representation": 0,
            "bits_stored": 8,
            "photometric_interpretation": "RGB",
        }
        buffer = encode_pixel_data(arr.tobytes(), **kwargs)
        assert np.array_equal(arr, decode(buffer))

    def test_photometric_interpretation(self):
        """Check photometric interpretation sets MCT correctly."""
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            arr = decode(f.read())

        kwargs = {
            "rows": 480,
            "columns": 640,
            "samples_per_pixel": 3,
            "pixel_representation": 0,
            "bits_stored": 8,
        }
        for pi in ("YBR_ICT", "YBR_RCT"):
            buffer = encode_pixel_data(
                arr.tobytes(),
                photometric_interpretation=pi,
                **kwargs,
            )
            param = parse_j2k(buffer)
            assert param["mct"] is True

        for pi in ("RGB", "YBR_FULL", "MONOCHROME1"):
            buffer = encode_pixel_data(
                arr.tobytes(),
                photometric_interpretation=pi,
                use_mct=True,
                **kwargs,
            )
            param = parse_j2k(buffer)
            assert param["mct"] is False

    def test_codec_format_ignored(self):
        """Test that codec_format gets ignored."""
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            arr = decode(f.read())

        kwargs = {
            "rows": 480,
            "columns": 640,
            "samples_per_pixel": 3,
            "pixel_representation": 0,
            "photometric_interpretation": "RGB",
            "bits_stored": 8,
            "codec_format": 1,
        }
        buffer = encode_pixel_data(arr.tobytes(), **kwargs)
        assert buffer.startswith(b"\xff\x4f\xff\x51")

    def test_pixel_representation(self):
        """Test pixel representation is applied correctly."""
        jpg = DIR_15444 / "HTJ2K" / "Bretagne1_ht.j2k"
        with open(jpg, "rb") as f:
            arr = decode(f.read())

        kwargs = {
            "rows": 480,
            "columns": 640,
            "samples_per_pixel": 3,
            "pixel_representation": 0,
            "photometric_interpretation": "RGB",
            "bits_stored": 8,
        }
        buffer = encode_pixel_data(arr.tobytes(), **kwargs)
        info = parse_j2k(buffer)
        assert info["is_signed"] is False

        kwargs["pixel_representation"] = 1
        buffer = encode_pixel_data(arr.tobytes(), **kwargs)
        info = parse_j2k(buffer)
        assert info["is_signed"] is True


class TestGetBitsStored:
    """Tests for _get_bits_stored()"""

    def check_signed(self, nr_bits, minimin, minimax, maximax, maximin):
        arr = np.asarray([minimin, minimax], dtype="<i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([minimax - 1, minimax], dtype="<i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([minimin, minimin + 1], dtype="<i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([maximin, maximax], dtype="<i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([0, maximax], dtype="<i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([minimin, 0], dtype="<i2")
        assert _get_bits_stored(arr) == nr_bits

    def test_bool(self):
        """Test bool input."""
        arr = np.asarray([0, 0], dtype="bool")
        assert _get_bits_stored(arr) == 1

        arr = np.asarray([1, 1], dtype="bool")
        assert _get_bits_stored(arr) == 1

    def test_unsigned(self):
        """Test unsigned integer input."""
        arr = np.asarray([0, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 1
        arr = np.asarray([1, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 1

        arr = np.asarray([2, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 2
        arr = np.asarray([3, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 2

        arr = np.asarray([4, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 3
        arr = np.asarray([7, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 3

        arr = np.asarray([8, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 4
        arr = np.asarray([15, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 4

        arr = np.asarray([16, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 5
        arr = np.asarray([31, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 5

        arr = np.asarray([32, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 6
        arr = np.asarray([63, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 6

        arr = np.asarray([64, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 7
        arr = np.asarray([127, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 7

        arr = np.asarray([128, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 8
        arr = np.asarray([255, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 8

        arr = np.asarray([256, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 9
        arr = np.asarray([511, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 9

        arr = np.asarray([512, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 10
        arr = np.asarray([1023, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 10

        arr = np.asarray([1024, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 11
        arr = np.asarray([2047, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 11

        arr = np.asarray([2048, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 12
        arr = np.asarray([4095, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 12

        arr = np.asarray([4096, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 13
        arr = np.asarray([8191, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 13

        arr = np.asarray([8192, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 14
        arr = np.asarray([16383, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 14

        arr = np.asarray([16384, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 15
        arr = np.asarray([32767, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 15

        arr = np.asarray([32768, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 16
        arr = np.asarray([65535, 0], dtype="<u2")
        assert _get_bits_stored(arr) == 16

    def test_signed(self):
        """Test signed integer input."""
        arr = np.asarray([0, 0], dtype="<i2")
        assert _get_bits_stored(arr) == 1
        arr = np.asarray([-1, 0], dtype="<i2")
        assert _get_bits_stored(arr) == 1

        minimin, minimax = -2, 1
        maximax, maximin = 1, -2
        self.check_signed(2, minimin, minimax, maximax, maximin)

        for ii in range(3, 17):
            minimin, minimax = maximin - 1, maximax + 1
            maximax, maximin = 2 ** (ii - 1) - 1, -(2 ** (ii - 1))
            self.check_signed(ii, minimin, minimax, maximax, maximin)
