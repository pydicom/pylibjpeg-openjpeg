
import os
from pathlib import Path
import shutil
from struct import unpack
import subprocess
from typing import List, Any


PACKAGE_DIR = Path(__file__).parent / "openjpeg"
LIB_DIR = Path(__file__).parent / "lib"
BUILD_TOOLS = Path(__file__).parent / "build_tools"
OPENJPEG_SRC = LIB_DIR / "openjpeg" / "src" / "lib" / "openjp2"
INTERFACE_SRC = LIB_DIR / "interface"
BUILD_DIR = LIB_DIR / "openjpeg" / "build"
BACKUP_DIR = BUILD_TOOLS / "backup"


def build(setup_kwargs: Any) -> Any:
    from setuptools import Extension
    from setuptools.dist import Distribution
    import Cython.Compiler.Options
    from Cython.Build import build_ext, cythonize
    import numpy

    setup_oj()

    # Determine if system is big endian or not
    macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    if unpack("h", b"\x00\x01")[0] == 1:
        macros.append(("PYOJ_BIG_ENDIAN", None))

    ext = Extension(
        "_openjpeg",
        [os.fspath(p) for p in get_source_files()],
        language="c",
        include_dirs=[
            os.fspath(OPENJPEG_SRC),
            os.fspath(INTERFACE_SRC),
            numpy.get_include(),
        ],
        extra_compile_args=[],
        extra_link_args=[],
        define_macros=macros,
    )

    ext_modules = cythonize(
        [ext],
        include_path=ext.include_dirs,
        language_level=3,
    )

    dist = Distribution({"ext_modules": ext_modules})
    cmd = build_ext(dist)
    cmd.ensure_finalized()
    cmd.run()

    for output in cmd.get_outputs():
        output = Path(output)
        relative_ext = output.relative_to(cmd.build_lib)
        shutil.copyfile(output, relative_ext)

    reset_oj()

    return setup_kwargs


def get_source_files() -> List[Path]:
    """Return a list of paths to the source files to be compiled."""
    source_files = [
        INTERFACE_SRC / "decode.c",
        INTERFACE_SRC / "encode.c",
        INTERFACE_SRC / "color.c",
        INTERFACE_SRC / "utils.c",
    ]
    for fname in OPENJPEG_SRC.glob("*"):
        if fname.parts[-1].startswith("test"):
            continue

        if fname.parts[-1].startswith("bench"):
            continue

        if fname.suffix == ".c":
            source_files.append(fname)

    source_files = [p.relative_to(Path(__file__).parent) for p in source_files]
    source_files.insert(0, PACKAGE_DIR / "_openjpeg.pyx")

    return source_files


def setup_oj() -> None:
    """Run custom cmake."""
    base_dir = LIB_DIR / "openjpeg"
    p_openjpeg = base_dir / "src" / "lib" / "openjp2" / "openjpeg.c"

    # Backup original CMakeLists.txt and openjpeg.c files
    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)

    BACKUP_DIR.mkdir(exist_ok=True, parents=True)

    shutil.copy(
        LIB_DIR / "openjpeg" / "CMakeLists.txt",
        BACKUP_DIR / "CMakeLists.txt.backup",
    )
    shutil.copy(
        OPENJPEG_SRC / "openjpeg.c",
        BACKUP_DIR / "openjpeg.c.backup",
    )

    # Copy custom CMakeLists.txt file to openjpeg base dir
    shutil.copy(
        BUILD_TOOLS / "cmake" / "CMakeLists.txt",
        LIB_DIR / "openjpeg" / "CMakeLists.txt",
    )
    # Edit openjpeg.c to remove the OPJ_API declaration
    with p_openjpeg.open("r") as f:
        data = f.readlines()

    data = [
        line.replace("OPJ_API ", "")
        if line.startswith("OPJ_API ") else line for line in data
    ]
    with p_openjpeg.open("w") as f:
        f.write("".join(data))

    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)

    try:
        os.remove(INTERFACE_SRC / "opj_config.h")
        os.remove(INTERFACE_SRC / "opj_config_private.h")
    except:
        pass

    os.mkdir(BUILD_DIR)
    cur_dir = os.getcwd()
    os.chdir(BUILD_DIR)
    subprocess.call(['cmake', os.fspath((LIB_DIR / "openjpeg").resolve(strict=True))])
    os.chdir(cur_dir)

    # Turn off JPIP
    if os.path.exists(INTERFACE_SRC / "opj_config.h"):
        with open(INTERFACE_SRC / "opj_config.h", "a") as f:
            f.write("\n")
            f.write("#define USE_JPIP 0")


def reset_oj() -> None:
    # Restore submodule to original state
    # Restore CMakeLists.txt and openjpeg.c files
    if (BACKUP_DIR / "CMakeLists.txt.backup").exists():
        shutil.copy(
            BACKUP_DIR / "CMakeLists.txt.backup",
            LIB_DIR / "openjpeg" / "CMakeLists.txt",
        )

    if (BACKUP_DIR / "openjpeg.c.backup").exists():
        shutil.copy(
            BACKUP_DIR / "openjpeg.c.backup",
            OPENJPEG_SRC / "openjpeg.c",
        )

    # Cleanup added directories
    if os.path.exists(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)

    if os.path.exists(BACKUP_DIR):
        shutil.rmtree(BACKUP_DIR)
