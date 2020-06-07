"""Unit tests for openjpeg."""

from io import BytesIO

from pydicom import dcmread
from pydicom.encaps import generate_pixel_data_frame
from pydicom.pixel_data_handlers.util import (
    get_expected_length, pixel_dtype, reshape_pixel_array
)
import matplotlib.pyplot as plt
import numpy as np
import pytest

import openjpeg
from openjpeg.utils import get_openjpeg_version, decode

#from openjpeg.data import get_indexed_datasets, JPEG_DIRECTORY


def test_version():
    """Test that the openjpeg version can be retrieved."""
    version = get_openjpeg_version()
    assert isinstance(version, tuple)
    assert isinstance(version[0], int)
    assert 3 == len(version)
    assert 2 == version[0]

def test_decode():
    #with open('')
    #arr = np.frombytes()
    ds = dcmread('MR2_J2KI.dcm')
    frame_gen = generate_pixel_data_frame(ds.PixelData, nr_frames=1)
    b = BytesIO(next(frame_gen))
    result, arr = decode(b, get_expected_length(ds, 'bytes'))

    arr = arr.view(pixel_dtype(ds))
    arr = reshape_pixel_array(ds, arr)
    plt.imshow(arr)
    plt.show()

#@pytest.mark.skip()
def test_decode_3s():
    #with open('')
    #arr = np.frombytes()
    ds = dcmread('SC.dcm')
    frame_gen = generate_pixel_data_frame(ds.PixelData, nr_frames=1)
    b = BytesIO(next(frame_gen))
    result, arr = decode(b, get_expected_length(ds, 'bytes'))

    arr = arr.view(pixel_dtype(ds))
    # Currently planar configuration 1, need to switch to 0
    arr = arr.reshape(ds.SamplesPerPixel, ds.Rows, ds.Columns)
    arr = arr.transpose(1, 2, 0)
    plt.imshow(arr)
    plt.show()
