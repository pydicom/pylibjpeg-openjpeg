
import os
import sys
from pathlib import Path
import shutil
import subprocess
from typing import List, Any


PACKAGE_DIR = Path(__file__).parent / "openjpeg"
BUILD_TOOLS = PACKAGE_DIR.parent / "build_tools"
OPENJPEG_SRC = PACKAGE_DIR / "src" / "openjpeg" / "src" / "lib" / "openjp2"
INTERFACE_SRC = PACKAGE_DIR / "src" / "interface"


def build(setup_kwargs: Any) -> Any:
    from setuptools import Extension
    from setuptools.dist import Distribution
    import Cython.Compiler.Options
    from Cython.Build import build_ext, cythonize
    import numpy

    setup_oj()

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
        INTERFACE_SRC / "color.c",
    ]
    for fname in OPENJPEG_SRC.glob("*"):
        if fname.parts[-1].startswith("test"):
            continue

        if fname.parts[-1].startswith("bench"):
            continue

        if fname.suffix == ".c":
            source_files.append(fname)

    source_files = [p.relative_to(PACKAGE_DIR.parent) for p in source_files]
    source_files.insert(0, Path("openjpeg/_openjpeg.pyx"))

    return source_files


def setup_oj() -> None:
    """Run custom cmake."""
    base_dir = PACKAGE_DIR / "src" / "openjpeg"
    p_openjpeg = base_dir / "src" / "lib" / "openjp2" / "openjpeg.c"

    # Backup original CMakeLists.txt and openjpeg.c files
    backup_dir = BUILD_TOOLS / "backup"
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)

    backup_dir.mkdir(exist_ok=True, parents=True)

    shutil.copy(
        base_dir / "CMakeLists.txt",
        backup_dir / "CMakeLists.txt.backup",
    )
    shutil.copy(
        p_openjpeg,
        backup_dir / "openjpeg.c.backup",
    )

    # Copy custom CMakeLists.txt file to openjpeg base dir
    shutil.copy(
        BUILD_TOOLS / "cmake" / "CMakeLists.txt",
        base_dir,
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

    build_dir = base_dir / "build"
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    try:
        os.remove(INTERFACE_SRC / "opj_config.h")
        os.remove(INTERFACE_SRC / "opj_config_private.h")
    except:
        pass

    os.mkdir(build_dir)
    fpath = os.path.abspath(base_dir)
    cur_dir = os.getcwd()
    os.chdir(build_dir)
    subprocess.call(['cmake', fpath])
    os.chdir(cur_dir)

    # Turn off JPIP
    if os.path.exists(INTERFACE_SRC / "opj_config.h"):
        with open(INTERFACE_SRC / "opj_config.h", "a") as f:
            f.write("\n")
            f.write("#define USE_JPIP 0")


def reset_oj() -> None:
    # Restore submodule to original state
    # Restore CMakeLists.txt and openjpeg.c files
    base_dir = PACKAGE_DIR / "src" / "openjpeg"
    build_dir = base_dir / "build"
    backup_dir = BUILD_TOOLS / "backup"
    p_openjpeg = base_dir / "src" / "lib" / "openjp2" / "openjpeg.c"

    if (backup_dir / "CMakeLists.txt.backup").exists():
        shutil.copy(
            backup_dir / "CMakeLists.txt.backup",
            base_dir / "CMakeLists.txt",
        )

    if (backup_dir / "openjpeg.c.backup").exists():
        shutil.copy(
            BUILD_TOOLS / "backup" / "openjpeg.c.backup",
            p_openjpeg,
        )

    # Cleanup added directories
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)

    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
