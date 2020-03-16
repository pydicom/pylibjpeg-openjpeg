

import _openjpeg


def get_openjpeg_version():
    """Return the openjpeg version."""
    version = _openjpeg.get_version()

    return version
