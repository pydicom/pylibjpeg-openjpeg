import sys

try:
    from . import pydicom_handler as handler
except ImportError:
    pass

# Add the testing data to openjpeg (if available)
try:
    import data as _data
    globals()['data'] = _data
    # Add to cache - needed for pytest
    sys.modules['openjpeg.data'] = _data
except ImportError:
    pass


def add_handler():
    """Add the pixel data handler to *pydicom*.
    Raises
    ------
    ImportError
        If *pydicom* is not available.
    """
    import pydicom.config

    if handler not in pydicom.config.pixel_data_handlers:
        pydicom.config.pixel_data_handlers.append(handler)


def remove_handler():
    """Remove the pixel data handler from *pydicom*.
    Raises
    ------
    ImportError
        If *pydicom* is not available.
    """
    import pydicom.config

    if handler in pydicom.config.pixel_data_handlers:
        pydicom.config.pixel_data_handlers.remove(handler)
