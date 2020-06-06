

import _openjpeg


def get_openjpeg_version():
    """Return the openjpeg version as tuple of int."""
    version = _openjpeg.get_version().decode('ascii').split('.')
    return tuple([int(ii) for ii in version])

def decode():
    return _openjpeg.opj_decode()
