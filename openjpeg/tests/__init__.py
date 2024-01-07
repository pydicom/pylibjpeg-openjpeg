import sys

# Add the testing data to openjpeg (if available)
try:
    import ljdata as _data

    globals()["data"] = _data
    # Add to cache - needed for pytest
    sys.modules["openjpeg.data"] = _data
except ImportError:
    pass
