
import math

import numpy as np
import pytest

from openjpeg import debug_logger
from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY
from openjpeg.utils import (
    encode,
    decode,
    get_parameters,
    PhotometricInterpretation as PI,
    _get_bits_stored,
)
from _openjpeg import encode as _encode


DIR_15444 = JPEG_DIRECTORY / "15444"


class TestCEncode:
    """Tests for _openjpeg.encode()"""

    def test_invalid_shape_raises(self):
        """Test invalid array shapes raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an invalid shape, "
            r"must be \(rows, columns\) or \(rows, columns, planes\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1,), dtype="u1"))

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 3, 4), dtype="u1"))

    def test_invalid_samples_per_pixel_raises(self):
        """Test invalid samples per pixel raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an unsupported number "
            "of samples per pixel, must be 1, 3 or 4"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 2), dtype="u1"))

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 5), dtype="u1"))

    def test_invalid_dtype_raises(self):
        """Test invalid array dtype raise exceptions."""
        msg = "input array has an unsupported dtype"
        for dtype in ("u8" ,"i8", "f", "d", "c", "U", "m", "M"):
            with pytest.raises((ValueError, RuntimeError), match=msg):
                encode(np.ones((1, 2), dtype=dtype))

    def test_invalid_contiguity_raises(self):
        """Test invalid array contiguity raise exceptions."""
        msg = (
            "Error encoding the data: the input array must be C-style, "
            "contiguous and aligned"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((3, 3), dtype="u1").T)

    def test_invalid_dimensions_raises(self):
        """Test invalid array dimensions raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an unsupported number "
            r"of rows, must be in \(1, 65535\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((65536, 1), dtype="u1"))

        msg = (
            "Error encoding the data: the input array has an unsupported number "
            r"of columns, must be in \(1, 65535\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 65536), dtype="u1"))

    def test_invalid_bits_stored_raises(self):
        """Test invalid bits_stored"""
        msg = (
            "Invalid value for the 'bits_stored' parameter, the value must "
            r"be in the range \(1, 8\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), bits_stored=0)

        with pytest.raises(ValueError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), bits_stored=9)

        msg = (
            "A 'bits_stored' value of 15 is incompatible with the range of "
            r"pixel data in the input array: \(-16385, 2\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode(np.asarray([-2**14 - 1, 2], dtype="i2"), bits_stored=15)

        msg = (
            "A 'bits_stored' value of 15 is incompatible with the range of "
            r"pixel data in the input array: \(-16384, 16384\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode(np.asarray([-2**14, 2**14], dtype="i2"), bits_stored=15)

        msg = (
            "A 'bits_stored' value of 4 is incompatible with the range of "
            r"pixel data in the input array: \(0, 16\)"
        )
        with pytest.raises(ValueError, match=msg):
            encode(np.asarray([0, 2**4], dtype="u2"), bits_stored=4)

    def test_invalid_pixel_value_raises(self):
        """Test invalid pixel values raise exceptions."""
        msg = (
            "The input array contains values outside the range of the maximum "
            "supported bit-depth of 24"
        )
        with pytest.raises(ValueError, match=msg):
            encode(np.asarray([2**24, 2], dtype="u4"))

        with pytest.raises(ValueError, match=msg):
            encode(np.asarray([2**23, 2], dtype="i4"))

        with pytest.raises(ValueError, match=msg):
            encode(np.asarray([-2**23 - 1, 2], dtype="i4"))

    def test_compression_snr_raises(self):
        """Test using compression_ratios and signal_noise_ratios raises."""
        msg = (
            "Only one of 'compression_ratios' or 'signal_noise_ratios' is "
            "allowed when performing lossy compression"
        )
        with pytest.raises(ValueError, match=msg):
            encode(
                np.asarray([0, 2], dtype="u2"),
                compression_ratios = [2, 1],
                signal_noise_ratios = [1, 2]
            )

    def test_invalid_photometric_raises(self):
        """Test invalid photometric_interpretation raises."""
        msg = (
            "Error encoding the data: the value of the 'photometric_interpretation' "
            "parameter is not valid for the number of samples per pixel in the "
            "input array"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), photometric_interpretation=PI.RGB)

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 3), dtype="u1"), photometric_interpretation=PI.MONOCHROME2)

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 4), dtype="u1"), photometric_interpretation=PI.MONOCHROME2)

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 4), dtype="u1"), photometric_interpretation=PI.RGB)

    def test_invalid_codec_format_raises(self):
        """Test an invalid 'codec_format' raises and exception."""
        msg = "The value of the 'codec_format' parameter is invalid, must be 0 or 2"
        with pytest.raises(ValueError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), codec_format=1)

    def test_invalid_compression_ratios_raises(self):
        """Test an invalid 'compression_ratios' raises exceptions."""
        msg = "More than 10 compression layers is not supported"
        with pytest.raises(ValueError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), compression_ratios=[1]*11)

        msg = (
            "Error encoding the data: invalid compression ratio, must be in the "
            r"range \(1, 1000\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), compression_ratios=[0])

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), compression_ratios=[1001])

    def test_invalid_signal_noise_ratios_raises(self):
        """Test an invalid 'signal_noise_ratios' raises exceptions."""
        msg = "More than 10 compression layers is not supported"
        with pytest.raises(ValueError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), signal_noise_ratios=[1]*11)

        msg = (
            "Error encoding the data: invalid signal-to-noise ratio, must be "
            r"in the range \(0, 1000\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), signal_noise_ratios=[-1])

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2), dtype="u1"), signal_noise_ratios=[1001])

    def test_encoding_failures_raise(self):
        """Miscellaneous test to check that failures are handled properly."""
        # Not exhaustive! Missing coverage for
        #   assigning image components
        #   creating the empty image
        #   setting the encoder
        #   creating the output stream
        #   opj_encode()
        #   opj_end_compress()

        # Input too small
        msg = r"Error encoding the data: failure result from 'opj_start_compress\(\)'"
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2), dtype="u1"))


class TestEncode:
    """Tests for openjpeg.encode()"""
    def test_lossless_bool(self):
        """Test encoding bool data for bit-depth 1"""
        rows = 123
        cols = 234
        planes = 3
        arr = np.random.randint(0, high=1, size=(rows, cols), dtype="bool")
        buffer = encode(arr, photometric_interpretation=PI.MONOCHROME2)
        out = decode(buffer)

        assert out.dtype.kind == "u"
        assert np.array_equal(arr, out)

        arr = np.random.randint(0, high=1, size=(rows, cols, planes), dtype="bool")
        buffer = encode(arr, photometric_interpretation=PI.RGB, use_mct=False)
        out = decode(buffer)

        assert out.dtype.kind == "u"
        assert np.array_equal(arr, out)

    def test_lossless_unsigned(self):
        """Test encoding unsigned data for bit-depth 1-16"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(1, 17):
            maximum = 2**bit_depth - 1
            dtype = f"u{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(0, high=maximum, size=(rows, cols), dtype=dtype)
            buffer = encode(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(0, high=maximum, size=(rows, cols, planes), dtype=dtype)
            buffer = encode(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_unsigned_u4(self):
        """Test encoding unsigned data for bit-depth 17-32"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(17, 25):
            maximum = 2**bit_depth - 1
            arr = np.random.randint(0, high=maximum, size=(rows, cols), dtype="u4")
            buffer = encode(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

            arr = np.random.randint(0, high=maximum, size=(rows, cols, planes), dtype="u4")
            buffer = encode(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            assert out.dtype.kind == "u"
            assert np.array_equal(arr, out)

    def test_lossless_signed_u4(self):
        """Test encoding signed data for bit-depth 17-32"""
        rows = 123
        cols = 234
        planes = 3
        for bit_depth in range(17, 25):
            maximum = 2**(bit_depth - 1) - 1
            minimum = -2**(bit_depth - 1)
            arr = np.random.randint(low=minimum, high=maximum, size=(rows, cols), dtype="i4")
            buffer = encode(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

            arr = np.random.randint(low=minimum, high=maximum, size=(rows, cols, planes), dtype="i4")
            buffer = encode(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_lossless_signed(self):
        """Test encoding signed data for bit-depth 1-16"""
        rows = 123
        cols = 543
        planes = 3
        for bit_depth in range(1, 17):
            maximum = 2**(bit_depth - 1) - 1
            minimum = -2**(bit_depth - 1)
            dtype = f"i{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(low=minimum, high=maximum, size=(rows, cols), dtype=dtype)
            buffer = encode(arr, photometric_interpretation=PI.MONOCHROME2)
            out = decode(buffer)

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

        for bit_depth in range(1, 17):
            maximum = 2**(bit_depth - 1) - 1
            minimum = -2**(bit_depth - 1)
            dtype = f"i{math.ceil(bit_depth / 8)}"
            arr = np.random.randint(low=minimum, high=maximum, size=(rows, cols, planes), dtype=dtype)
            buffer = encode(arr, photometric_interpretation=PI.RGB, use_mct=False)
            out = decode(buffer)

            assert out.dtype.kind == "i"
            assert np.array_equal(arr, out)

    def test_encode_u1_ybr(self):
        """Test some existing J2K files."""
        print("Raw length", 256 * 256 * 3)

        jpg = DIR_15444 / "2KLS" / "oj36.j2k"
        with open(jpg, "rb") as f:
            data = f.read()
            print("Original compressed length", len(data))

        arr = decode(data)
        assert "uint8" == arr.dtype
        assert (256, 256, 3) == arr.shape

        # Test lossless
        result = encode(arr, photometric_interpretation=PI.YBR_FULL)
        print("Lossless length", len(result))
        out = decode(result)
        assert np.array_equal(arr, out)

        # Test lossy
        result = encode(
            arr,
            photometric_interpretation=PI.YBR_FULL,
            lossless=0,
            compression_ratios = [6, 4, 2, 1]
        )
        print("Lossy length", len(result))
        out = decode(result)
        diff = arr.astype("float") - out.astype("float")
        assert diff.max() <= 2
        assert diff.min() >= -2

    def test_encode_i16_mono(self):
        print("Raw length", 512 * 512 * 2)

        jpg = DIR_15444 / "2KLS" / "693.j2k"
        with open(jpg, "rb") as f:
            data = f.read()
            print("Original compressed length", len(data))

        import tifffile

        arr = decode(data)
        tifffile.imwrite("test_i2.tif", arr.astype("i2"))
        assert "i2" == arr.dtype
        assert (512, 512) == arr.shape

        # Test lossless
        result = encode(
            arr,
            photometric_interpretation=PI.MONOCHROME2,
        )
        # 105389 with precision 16, 105383 with 13
        print("Lossless length", len(result))
        out = decode(result)
        lossless = out
        assert np.array_equal(arr, out)

        # Test lossy w/ compression ratios
        print("CR")
        result = encode(
            arr,
            photometric_interpretation=PI.MONOCHROME2,
            compression_ratios = [6, 4, 2, 1]
        )
        print(arr.max(), arr.min())
        # min/max is -2000/2492 -> 13-bit
        # Weirdly is 93652 bytes if precision is 13, but 93589 for 16
        print("Lossy length", len(result))
        out = decode(result)
        diff = arr.astype("float") - out.astype("float")
        assert diff.max() <= 2
        assert diff.min() >= -2

        # Test lossy w/ signal-to-noise ratios
        print("SNR")
        result = encode(
            arr,
            photometric_interpretation=PI.MONOCHROME2,
            signal_noise_ratios = [20, 30, 40, 50]
        )
        print(arr.max(), arr.min())
        # min/max is -2000/2492 -> 13-bit
        # Weirdly is 93652 bytes if precision is 13, but 93589 for 16
        print("Lossy length", len(result))
        out = decode(result)
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("TkAgg")
        plot, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(arr)
        ax2.imshow(lossless)
        ax3.imshow(out)
        plt.show()
        diff = arr.astype("float") - out.astype("float")
        print(out.max(), out.min())
        print(diff.max(), diff.min())
        assert diff.max() <= 2
        assert diff.min() >= -2


class TestGetBitsStored:
    """Tests for _get_bits_stored()"""
    def check_signed(self, nr_bits, minimin, minimax, maximax, maximin):
        arr = np.asarray([minimin, minimax], dtype="i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([minimax - 1, minimax], dtype="i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([minimin, minimin + 1], dtype="i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([maximin, maximax], dtype="i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([0, maximax], dtype="i2")
        assert _get_bits_stored(arr) == nr_bits
        arr = np.asarray([minimin, 0], dtype="i2")
        assert _get_bits_stored(arr) == nr_bits

    def test_bool(self):
        """Test bool input."""
        arr = np.asarray([0, 0], dtype="bool")
        assert _get_bits_stored(arr) == 1

        arr = np.asarray([1, 1], dtype="bool")
        assert _get_bits_stored(arr) == 1

    def test_unsigned(self):
        """Test unsigned integer input."""
        arr = np.asarray([0, 0], dtype="u2")
        assert _get_bits_stored(arr) == 1
        arr = np.asarray([1, 0], dtype="u2")
        assert _get_bits_stored(arr) == 1

        arr = np.asarray([2, 0], dtype="u2")
        assert _get_bits_stored(arr) == 2
        arr = np.asarray([3, 0], dtype="u2")
        assert _get_bits_stored(arr) == 2

        arr = np.asarray([4, 0], dtype="u2")
        assert _get_bits_stored(arr) == 3
        arr = np.asarray([7, 0], dtype="u2")
        assert _get_bits_stored(arr) == 3

        arr = np.asarray([8, 0], dtype="u2")
        assert _get_bits_stored(arr) == 4
        arr = np.asarray([15, 0], dtype="u2")
        assert _get_bits_stored(arr) == 4

        arr = np.asarray([16, 0], dtype="u2")
        assert _get_bits_stored(arr) == 5
        arr = np.asarray([31, 0], dtype="u2")
        assert _get_bits_stored(arr) == 5

        arr = np.asarray([32, 0], dtype="u2")
        assert _get_bits_stored(arr) == 6
        arr = np.asarray([63, 0], dtype="u2")
        assert _get_bits_stored(arr) == 6

        arr = np.asarray([64, 0], dtype="u2")
        assert _get_bits_stored(arr) == 7
        arr = np.asarray([127, 0], dtype="u2")
        assert _get_bits_stored(arr) == 7

        arr = np.asarray([128, 0], dtype="u2")
        assert _get_bits_stored(arr) == 8
        arr = np.asarray([255, 0], dtype="u2")
        assert _get_bits_stored(arr) == 8

        arr = np.asarray([256, 0], dtype="u2")
        assert _get_bits_stored(arr) == 9
        arr = np.asarray([511, 0], dtype="u2")
        assert _get_bits_stored(arr) == 9

        arr = np.asarray([512, 0], dtype="u2")
        assert _get_bits_stored(arr) == 10
        arr = np.asarray([1023, 0], dtype="u2")
        assert _get_bits_stored(arr) == 10

        arr = np.asarray([1024, 0], dtype="u2")
        assert _get_bits_stored(arr) == 11
        arr = np.asarray([2047, 0], dtype="u2")
        assert _get_bits_stored(arr) == 11

        arr = np.asarray([2048, 0], dtype="u2")
        assert _get_bits_stored(arr) == 12
        arr = np.asarray([4095, 0], dtype="u2")
        assert _get_bits_stored(arr) == 12

        arr = np.asarray([4096, 0], dtype="u2")
        assert _get_bits_stored(arr) == 13
        arr = np.asarray([8191, 0], dtype="u2")
        assert _get_bits_stored(arr) == 13

        arr = np.asarray([8192, 0], dtype="u2")
        assert _get_bits_stored(arr) == 14
        arr = np.asarray([16383, 0], dtype="u2")
        assert _get_bits_stored(arr) == 14

        arr = np.asarray([16384, 0], dtype="u2")
        assert _get_bits_stored(arr) == 15
        arr = np.asarray([32767, 0], dtype="u2")
        assert _get_bits_stored(arr) == 15

        arr = np.asarray([32768, 0], dtype="u2")
        assert _get_bits_stored(arr) == 16
        arr = np.asarray([65535, 0], dtype="u2")
        assert _get_bits_stored(arr) == 16

    def test_signed(self):
        """Test signed integer input."""
        arr = np.asarray([0, 0], dtype="i2")
        assert _get_bits_stored(arr) == 1
        arr = np.asarray([-1, 0], dtype="i2")
        assert _get_bits_stored(arr) == 1

        minimin, minimax = -2, 1
        maximax, maximin = 1, -2
        self.check_signed(2, minimin, minimax, maximax, maximin)

        for ii in range(3, 17):
            minimin, minimax = maximin - 1, maximax + 1
            maximax, maximin = 2**(ii - 1) - 1, -2**(ii - 1)
            self.check_signed(ii, minimin, minimax, maximax, maximin)
