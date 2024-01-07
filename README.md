<p align="center">
<a href="https://github.com/pydicom/pylibjpeg-openjpeg/actions?query=workflow%3Aunit-tests"><img alt="Build status" src="https://github.com/pydicom/pylibjpeg-openjpeg/workflows/unit-tests/badge.svg"></a>
<a href="https://codecov.io/gh/pydicom/pylibjpeg-openjpeg"><img alt="Test coverage" src="https://codecov.io/gh/pydicom/pylibjpeg-openjpeg/branch/main/graph/badge.svg"></a>
<a href="https://pypi.org/project/pylibjpeg-openjpeg/"><img alt="PyPI versions" src="https://img.shields.io/pypi/v/pylibjpeg-openjpeg"></a>
<a href="https://www.python.org/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/pylibjpeg-openjpeg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


## pylibjpeg-openjpeg

A Python 3.8+ wrapper for
[openjpeg](https://github.com/uclouvain/openjpeg), with a focus on use as a plugin for [pylibjpeg](http://github.com/pydicom/pylibjpeg).

Linux, OSX and Windows are all supported.

### Installation
#### Dependencies
[NumPy](http://numpy.org)

#### Installing the current release
```bash
python -m pip install -U pylibjpeg-openjpeg
```

#### Installing the development version

Make sure [Python](https://www.python.org/), [Git](https://git-scm.com/) and [CMake](https://cmake.org/) are installed. For Windows, you also need to install
[Microsoft's C++ Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
```bash
git clone --recurse-submodules https://github.com/pydicom/pylibjpeg-openjpeg
python -m pip install pylibjpeg-openjpeg
```


### Supported JPEG Formats
#### Decoding

| ISO/IEC Standard | ITU Equivalent | JPEG Format |
| --- | --- | --- |
| [15444-1](https://www.iso.org/standard/78321.html) | [T.800](https://www.itu.int/rec/T-REC-T.800/en) | [JPEG 2000](https://jpeg.org/jpeg2000/) |

#### Encoding
Encoding of JPEG 2000 images is not currently supported


### Transfer Syntaxes
| UID | Description |
| --- | --- |
| 1.2.840.10008.1.2.4.90 | JPEG 2000 Image Compression (Lossless Only) |
| 1.2.840.10008.1.2.4.91 | JPEG 2000 Image Compression |
| 1.2.840.10008.1.2.4.201 | High-Throughput JPEG 2000 Image Compression (Lossless Only) |
| 1.2.840.10008.1.2.4.202 | High-Throughput JPEG 2000 with RPCL Options Image Compression (Lossless Only) |
| 1.2.840.10008.1.2.4.203 | High-Throughput JPEG 2000 Image Compression |


### Usage
#### With pylibjpeg and pydicom

```python
from pydicom import dcmread
from pydicom.data import get_testdata_file

ds = dcmread(get_testdata_file('JPEG2000.dcm'))
arr = ds.pixel_array
```

#### Standalone JPEG decoding

You can also decode JPEG 2000 images to a [numpy ndarray][1]:

[1]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

```python
from openjpeg import decode

with open('filename.j2k', 'rb') as f:
    # Returns a numpy array
    arr = decode(f)

# Or simply...
arr = decode('filename.j2k')
```
