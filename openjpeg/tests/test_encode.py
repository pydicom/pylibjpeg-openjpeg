
import numpy as np
import pytest

from openjpeg.utils import encode, decode


class TestEncode:
    """Tests for encode()"""

    def test_invalid_shape_raises(self):
        """Test invalid array shapes raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an invalid shape, "
            r"must be \(rows, columns\) or \(rows, columns, planes\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1,)))

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 3, 4)))

    def test_invalid_samples_per_pixel_raises(self):
        """Test invalid samples per pixel raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an invalid number of "
            "samples per pixel, must be 1, 3 or 4"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 2)))

        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 2, 5)))

    def test_invalid_dtype_raises(self):
        """Test invalid array dtype raise exceptions."""
        msg = (
            "Error encoding the data: the input array has an invalid dtype, "
            "only bool, u1, u2, i1 and i2 are supported"
        )
        for dtype in ("u4", "i4", "u8" ,"i8", "f", "d", "c", "U", "m", "M"):
            with pytest.raises(RuntimeError, match=msg):
                encode(np.ones((1, 2), dtype=dtype))

    # def test_invalid_endianness_raises(self):
    #     """Test invalid array endianness raise exceptions."""
    #     msg = (
    #         "Error encoding the data: the input array must use little endian "
    #         "byte ordering"
    #     )
    #     with pytest.raises(RuntimeError, match=msg):
    #         encode(np.ones((2, 2), dtype=">u2"))

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
            "Error encoding the data: the input array has an invalid shape, "
            r"the number of rows must be in \(1, 65535\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((65536, 1)))

        msg = (
            "Error encoding the data: the input array has an invalid shape, "
            r"the number of columns must be in \(1, 65535\)"
        )
        with pytest.raises(RuntimeError, match=msg):
            encode(np.ones((1, 65536)))


def test_encodes():
    from pydicom.data import get_testdata_file
    ds = get_testdata_file("CT_small.dcm", read=True)
    arr = ds.pixel_array
    print(arr.shape, arr.dtype)
    result = encode(arr, bits_stored=16, photometric_interpretation=2, use_mct=0)
    print(result[0], len(result[1]))
    # with open("test.j2k", "wb") as f:
    #     f.write(result[1])

    out = decode(result[1])
    print("Original", arr)
    print("After encode + decode", out)
    print("Equal?", np.array_equal(arr, out))

    import matplotlib.pyplot as plt
    plot, (ax1, ax2) = plt.subplots(2)
    ax1.imshow(arr)
    ax2.imshow(out)
    plt.show()


@pytest.mark.skip()
def test_encode():
    print("\nbool, 1, 0, 0, 1, 1, 0")
    data = [[1, 0], [0, 1], [1, 0]]
    arr = np.asarray(data, dtype="bool")
    encode(arr)

    print("\nu1: 1, 255, 3, 4, 5, 6")
    data = [[1, 255], [3, 4], [5, 6]]
    arr = np.asarray(data, dtype="u1")
    encode(arr)

    print("\nu2: 1012, 2, 3233, 4, 512, 6")
    data = [[1012, 2], [3233, 4], [512, 6]]
    arr = np.asarray(data, dtype="u2")
    encode(arr)

    print("\ni1: -1, 2, 3, -4, -5, 6")
    data = [[-1, 2], [3, -4], [-5, 6]]
    arr = np.asarray(data, dtype="i1")
    encode(arr)

    print("\ni1: ")
    # print("i1, 3 samples per pixel")
    data = (
        [
            [ # row 1
                [-1, 2, 3],
                [1, -2, -3],
                [1, -2, -3],
                [1, -2, -3],
            ],
            [ # row 2
                [-1, 2, 3],
                [1, -2, -3],
                [1, -2, -3],
                [1, -2, -3],
            ],
        ]
    )
    arr = np.asarray(data, dtype="i1")
    encode(arr)

    print("\ni2: -1012, 2, 3233, -4, -512, 6")
    data = [[-1012, 2], [3233, -4], [-512, 6]]
    arr = np.asarray(data, dtype="i2")
    encode(arr)

    # print("2 x 3, n1, 16-bit signed")
    # arr = np.empty((2, 3), dtype="i2")
    # encode(arr)
    # print("2 x 3, n2, 8-bit signed")
    # arr = np.empty((2, 3, 4), dtype="i1")
    # encode(arr)
    # print("2 x 3, n2, 16-bit signed")
    # arr = np.empty((2, 3, 4), dtype="i2")
    # encode(arr)
    # print("2 x 3, n1, 8-bit unsigned")
    # arr = np.empty((2, 3), dtype="u1")
    # encode(arr)
    # print("2 x 3, n1, 16-bit unsigned")
    # arr = np.empty((2, 3), dtype="u2")
    # encode(arr)
    # print("2 x 3, n2, 8-bit unsigned")
    # arr = np.empty((2, 3, 4), dtype="u1")
    # encode(arr)
    # print("2 x 3, n2, 16-bit unsigned")
    # arr = np.empty((2, 3, 4), dtype="u2")
    # encode(arr)
    # print("2 x 3, n1, bool")
    # arr = np.empty((2, 3), dtype="bool")
    # encode(arr)
    # print("2 x 3, n2, bool")
    # arr = np.empty((2, 3, 4), dtype="bool")
    # encode(arr)
    # print("1, n1, u1")
    # arr = np.empty((10,), dtype="u1")
    # encode(arr)
    # print("1 x 2, n5, n1, u1")
    # arr = np.empty((1, 2, 5), dtype="u1")
    # encode(arr)
    # print("1 x 2 x 3 x 4, n1, u1")
    # arr = np.empty((1, 2, 3, 4), dtype="u1")
    # encode(arr)
