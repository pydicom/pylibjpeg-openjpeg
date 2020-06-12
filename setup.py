
import os
import sys
from pathlib import Path
import platform
import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension
import shutil
import subprocess
from distutils.command.build import build as build_orig
import distutils.sysconfig


OPENJPEG_SRC = os.path.join(
    "openjpeg", "src", "openjpeg", "src", "lib", "openjp2"
)
INTERFACE_SRC = os.path.join("openjpeg", "src", "interface")


# Workaround for needing cython and numpy
# Solution from: https://stackoverflow.com/a/54128391/12606901
class build(build_orig):
    def finalize_options(self):
        super().finalize_options()
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        extension = next(
            m for m in self.distribution.ext_modules if m == ext
        )
        extension.include_dirs.append(numpy.get_include())


def get_source_files():
    """Return a list of paths to the source files to be compiled."""
    source_files = [
        "openjpeg/_openjpeg.pyx",
        os.path.join(INTERFACE_SRC, "decode.c"),
    ]
    for fname in Path(OPENJPEG_SRC).glob("*"):
        if fname.parts[-1].startswith("test"):
            continue

        if fname.parts[-1].startswith("bench"):
            continue

        fname = str(fname)
        if fname.endswith(".c"):
            source_files.append(fname)

    return source_files


def setup_oj():
    """Run custom cmake."""
    base_dir = os.path.join("openjpeg", "src", "openjpeg")

    # Copy custom CMakeLists.txt file to openjpeg base dir
    shutil.copy(
        os.path.join("build_tools", "cmake", "CMakeLists.txt"),
        base_dir
    )
    build_dir = os.path.join(base_dir, "build")
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    try:
        os.remove(os.path.join(INTERFACE_SRC, "opj_config.h"))
        os.remove(os.path.join(INTERFACE_SRC, "opj_config_private.h"))
    except:
        pass

    os.mkdir(build_dir)
    fpath = os.path.abspath(base_dir)
    cur_dir = os.getcwd()
    os.chdir(build_dir)
    subprocess.call(['cmake', fpath])
    os.chdir(cur_dir)

    # Turn off JPIP
    if os.path.exists(os.path.join(INTERFACE_SRC, "opj_config.h")):
        with open(os.path.join(INTERFACE_SRC, "opj_config.h"), "a") as f:
            f.write("\n")
            f.write("#define USE_JPIP 0")


setup_oj()


# Compiler and linker arguments
extra_compile_args = []
extra_link_args = []

# Maybe use cythonize instead
ext = Extension(
    "_openjpeg",
    get_source_files(),
    language="c",
    include_dirs=[
        OPENJPEG_SRC,
        INTERFACE_SRC,
        distutils.sysconfig.get_python_inc(),
        # Numpy includes get added by the `build` subclass
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# Version
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VERSION_FILE = os.path.join(BASE_DIR, "openjpeg", "_version.py")
with open(VERSION_FILE) as fp:
    exec(fp.read())

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name = "pylibjpeg-openjpeg",
    packages = find_packages(),
    include_package_data = True,
    version = __version__,
    zip_safe = False,
    description = (
        "A Python wrapper for openjpeg, with a focus on use as a plugin for "
        "for pylibjpeg"
    ),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    author = "scaramallion",
    author_email = "scaramallion@users.noreply.github.com",
    url = "https://github.com/scaramallion/pylibjpeg-openjpeg",
    license = "MIT",
    keywords = (
        "dicom python medicalimaging radiotherapy oncology pydicom imaging "
        "jpg jpg2000 jpeg jpeg2000 pylibjpeg openjpeg"
    ),
    classifiers = [
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: BSD License",
        "License :: OSI Approved :: Historical Permission Notice and Disclaimer (HPND)",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires = ">=3.6",
    setup_requires = ["setuptools>=18.0", "cython", "numpy"],
    install_requires = ["numpy"],
    cmdclass = {"build": build},
    ext_modules = [ext],
    # Plugin registrations
    entry_points = {
        "pylibjpeg.pixel_data_decoders": [
            "1.2.840.10008.1.2.4.90 = openjpeg:decode_pixel_data",
            "1.2.840.10008.1.2.4.91 = openjpeg:decode_pixel_data",
        ],
        "pylibjpeg.jpeg_2000_decoders": "openjpeg = openjpeg:decode",
    }
)
