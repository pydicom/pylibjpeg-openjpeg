"""Set package shortcuts."""

from ._version import __version__


def _dinstall(decoder):
    """Install the JPEG `decoder` in the pixel data handler.

    Parameters
    ----------
    decoder : callable
        The decoder function to be installed.
    """
    from pyjpeg.libjpeg_handler import _DECODERS, _LIBJPEG_TRANSFER_SYNTAXES
    for tsyntax in _LIBJPEG_TRANSFER_SYNTAXES:
        _DECODERS[tsyntax] = decoder


def _einstall(encoder):
    """Install the JPEG `encoder` in the pixel data handler.

    Parameters
    ----------
    encoder : callable
        The encoder function to be installed.
    """
    from pyjpeg.libjpeg_handler import _ENCODERS, _LIBJPEG_TRANSFER_SYNTAXES
    for tsyntax in _LIBJPEG_TRANSFER_SYNTAXES:
        _ENCODERS[tsyntax] = encoder


try:
    import pylibjpeg
    # `install_decoder` and `install_encoder` are MIT licensed
    pylibjpeg.install_decoder(_dinstall)
    pylibjpeg.install_encoder(_einstall)
except ImportError:
    pass
