"""Set package shortcuts."""

import sys

from ._version import __version__
#from .utils import decode


# Add the testing data to openjpeg (if available)
try:
    import data as _data
    globals()['data'] = _data
    # Add to cache - needed for pytest
    sys.modules['openjpeg.data'] = _data
except ImportError:
    pass
