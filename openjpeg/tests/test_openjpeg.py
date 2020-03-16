import pytest

import openjpeg
from openjpeg.utils import get_openjpeg_version

def test_version():
    print(get_openjpeg_version())
