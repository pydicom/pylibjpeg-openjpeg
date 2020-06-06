"""Unit tests for openjpeg."""

import pytest

import openjpeg
from openjpeg.utils import get_openjpeg_version, decode


def test_version():
    """Test that the openjpeg version can be retrieved."""
    version = get_openjpeg_version()
    assert isinstance(version, tuple)
    assert isinstance(version[0], int)
    assert 3 == len(version)
    assert 2 == version[0]


def test_decode():
    print(decode())
