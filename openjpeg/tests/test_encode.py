
import numpy as np
import pytest

from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY
from openjpeg.utils import encode, decode, PhotometricInterpretation as PI


DIR_15444 = JPEG_DIRECTORY / "15444"


class TestCEncode:
    """Tests for _openjpeg.Encode()"""

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
        for dtype in ("u4", "i4", "u8" ,"i8", "f", "d", "c", "U", "m", "M"):
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


class TestEncode:
    """Tests for openjpeg.encode()"""
    def test_encode(self):
        print("Raw length", 256 * 256 * 3)

        jpg = DIR_15444 / "2KLS" / "oj36.j2k"
        with open(jpg, "rb") as f:
            data = f.read()
            print("Original compressed length", len(data))

        arr = decode(data)
        assert "uint8" == arr.dtype
        assert (256, 256, 3) == arr.shape
        assert [235, 244, 245] == arr[0, 0, :].tolist()

        # Test lossless
        result = encode(arr, bits_stored=8, photometric_interpretation=PI.RGB)
        print("Lossless length", len(result))
        out = decode(result)
        assert np.array_equal(arr, out)

        # Test lossy
        result = encode(
            arr,
            bits_stored=8,
            photometric_interpretation=PI.RGB,
            lossless=0,
            compression_ratios = [6, 4, 2, 1]
        )
        print("Lossy length", len(result))
        out = decode(result)
        diff = arr.astype("float") - out.astype("float")
        assert diff.max() == 2
        assert diff.min() == -2
