## pylibjpeg-openjpeg

A Python 3.6+ wrapper for
[openjpeg](https://github.com/uclouvain/openjpeg), with a focus on use as a
plugin for [pylibjpeg](http://github.com/pydicom/pylibjpeg).

Linux, OSX and Windows will all be supported.

### Installation
#### Dependencies
[NumPy](http://numpy.org)

#### Installing the current release
Not yet available

#### Installing the development version

Make sure [Python](https://www.python.org/) and [Git](https://git-scm.com/) are installed. For Windows, you also need to install
[Microsoft's C++ Build Tools](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16).
```bash
git clone --recurse-submodules https://github.com/scaramallion/pylibjpeg-openjpeg
python -m pip install pylibjpeg-openjpeg
```


### Supported JPEG Formats
#### Decoding

| ISO/IEC Standard | ITU Equivalent | JPEG Format |
| --- | --- | --- |
| [15444-1](https://www.iso.org/standard/78321.html)   | [T.800](https://www.itu.int/rec/T-REC-T.800/en) | [JPEG 2000](https://jpeg.org/jpeg2000/) |

#### Encoding
Encoding of JPEG 2000 images is not currently supported


### Transfer Syntaxes
| UID | Description |
| --- | --- |
| 1.2.840.10008.1.2.4.90 | JPEG 2000 Image Compression (Lossless Only) |
| 1.2.840.10008.1.2.4.91 | JPEG 2000 Image Compression |


### Usage
#### With pylibjpeg and pydicom

```python
from pydicom import dcmread
from pydicom.data import get_testdata_file

import pylibjpeg

# Doesn't work yet
ds = dcmread(get_testdata_file('JPEG2000.dcm'))
arr = ds.pixel_array
```

#### Standalone JPEG decoding

You can also decode JPEG 2000 images to a [numpy ndarray][1]:

[1]: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html

```python
from openjpeg import decode

with open('filename.jpg', 'rb') as f:
    # Returns a numpy array
    arr = decode(f.read())
```
