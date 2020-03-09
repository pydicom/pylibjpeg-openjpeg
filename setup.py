

import os
import sys
from pathlib import Path
import platform
import setuptools
from setuptools import setup, find_packages
from setuptools.extension import Extension
import subprocess
from distutils.command.build import build as build_orig
import distutils.sysconfig


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

# Maybe use cythonize instead
ext = Extension(
    '_openjpeg',
    source_files,
    language='c',
    include_dirs=include_dirs,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

# Version
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VERSION_FILE = os.path.join(BASE_DIR, 'openjpeg', '_version.py')
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
    long_description_content_type = 'text/markdown',
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
        "Development Status :: 1 - Planning",
        #"Development Status :: 2 - Pre-Alpha",
        #"Development Status :: 3 - Alpha",
        #"Development Status :: 4 - Beta",
        #"Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires = ">=3.6",
    setup_requires = ['setuptools>=18.0', 'cython', 'numpy'],
    install_requires = ["numpy"],
    cmdclass = {'build': build},
    ext_modules = [ext],
)
