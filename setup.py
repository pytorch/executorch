# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Part of this code is from pybind11 cmake_example:
# https://github.com/pybind/cmake_example/blob/master/setup.py so attach the
# license below.

# Copyright (c) 2016 The Pybind Development Team, All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.

import contextlib

# Import this before distutils so that setuptools can intercept the distuils
# imports.
import importlib.util
import logging
import os
import re
import shutil
import site
import subprocess
import sys
from distutils import log  # type: ignore[import-not-found]
from distutils.sysconfig import get_python_lib  # type: ignore[import-not-found]
from pathlib import Path
from typing import List, Optional

# Clean dynamic import using importlib
_install_utils_path = Path(__file__).parent / "install_utils.py"
_spec = importlib.util.spec_from_file_location("install_utils", _install_utils_path)
if _spec is None:
    raise ImportError(f"Could not create module spec for {_install_utils_path}")
install_utils = importlib.util.module_from_spec(_spec)
if _spec.loader is None:
    raise ImportError(f"Module spec has no loader for {_install_utils_path}")
_spec.loader.exec_module(install_utils)

from setuptools import Extension, find_namespace_packages, setup
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

try:
    from tools.cmake.cmake_cache import CMakeCache
except ImportError:
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "cmake")
    )
    from cmake_cache import CMakeCache  # type: ignore[no-redef, import-not-found]


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().upper() in {"1", "ON", "TRUE", "YES"}


def _is_minimal_build() -> bool:
    return _is_env_flag_enabled("EXECUTORCH_BUILD_MINIMAL")


def _minimal_cmake_flags() -> List[str]:
    return [
        "-DEXECUTORCH_BUILD_COREML=OFF",
        "-DEXECUTORCH_BUILD_CUDA=OFF",
        "-DEXECUTORCH_BUILD_DEVTOOLS=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_MODULE=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_TENSOR=OFF",
        "-DEXECUTORCH_BUILD_EXTENSION_TRAINING=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_CUSTOM_AOT=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_LLM=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_LLM_AOT=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED=OFF",
        "-DEXECUTORCH_BUILD_KERNELS_QUANTIZED_AOT=OFF",
        "-DEXECUTORCH_BUILD_MLX=OFF",
        "-DEXECUTORCH_BUILD_OPENVINO=OFF",
        "-DEXECUTORCH_BUILD_PORTABLE_OPS=OFF",
        "-DEXECUTORCH_BUILD_PYBIND=OFF",
        "-DEXECUTORCH_BUILD_QNN=OFF",
        "-DEXECUTORCH_BUILD_TESTS=OFF",
        "-DEXECUTORCH_BUILD_XNNPACK=OFF",
    ]


def _minimal_packages() -> List[str]:
    return sorted(
        find_namespace_packages(
            where="src",
            include=[
                "executorch",
                "executorch.data",
                "executorch.data.bin",
                "executorch.exir",
                "executorch.exir.*",
                "executorch.extension",
                "executorch.extension.flat_tensor",
                "executorch.extension.flat_tensor.*",
                "executorch.extension.pytree",
            ],
            exclude=[
                "*.test",
                "*.test.*",
                "*.tests",
                "*.tests.*",
                "*.__pycache__",
                "*.__pycache__.*",
            ],
        )
    )


def _base_dependencies() -> List[str]:
    """Runtime dependencies for the full wheel.

    Declared here rather than in pyproject.toml (where `dependencies` is marked
    dynamic) so the minimal build can ship a slimmer set. Keep in sync with the
    project's runtime needs.
    """
    return [
        "expecttest",
        "flatbuffers",
        "hypothesis",
        "kgb",
        "mpmath==1.3.0",
        "numpy>=2.0.0; python_version >= '3.10'",
        "packaging",
        "pandas>=2.2.2; python_version >= '3.10'",
        "parameterized",
        "pytorch-tokenizers",
        "pyyaml",
        "ruamel.yaml",
        "sympy",
        "tabulate",
        # See also third-party/TARGETS for buck's typing-extensions version.
        "typing-extensions>=4.10.0",
        # Keep this version in sync with: ./backends/apple/coreml/scripts/install_requirements.sh
        "coremltools==9.0; platform_system == 'Darwin' or platform_system == 'Linux'",
        # scikit-learn is used to support palettization in the coreml backend.
        "scikit-learn==1.7.1",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
    ]


def _minimal_dependencies() -> List[str]:
    """Runtime dependencies for the minimal (AOT export only) wheel.

    Derived as the subset of _base_dependencies() that executorch.exir needs to
    lower and serialize a .pte, so version pins and markers stay in sync with the
    full set. torch is intentionally absent from both (consumers bring their own).
    mpmath is intentionally dropped too: it is pulled transitively by sympy, whose
    "mpmath<1.4" cap resolves to the same 1.3.0 the full wheel pins. Keep the name
    set below in sync with the `expected` set in .ci/scripts/test_minimal_wheel.sh.
    """
    keep = {
        "flatbuffers",
        "numpy",
        "packaging",
        "pyyaml",
        "ruamel-yaml",
        "sympy",
        "tabulate",
        "typing-extensions",
    }

    def _name(dep: str) -> str:
        # PEP 503 normalized distribution name, e.g. "ruamel.yaml" -> "ruamel-yaml".
        return re.sub(
            r"[-_.]+", "-", re.split(r"[ ;\[<>=!~(]", dep, maxsplit=1)[0]
        ).lower()

    minimal = [dep for dep in _base_dependencies() if _name(dep) in keep]
    # Fail the build loudly if a name in `keep` no longer matches a full-wheel dep
    # (e.g. renamed or removed in _base_dependencies()), instead of silently
    # shipping a minimal wheel that is missing a required dependency.
    unmatched = keep - {_name(dep) for dep in minimal}
    assert not unmatched, f"minimal keep-set names not found in base deps: {unmatched}"
    return minimal


class Version:
    """Static strings that describe the version of the pip package."""

    # Cached values returned by the properties.
    __root_dir_attr: Optional[str] = None
    __string_attr: Optional[str] = None
    __git_hash_attr: Optional[str] = None

    @classmethod
    def _root_dir(cls) -> str:
        """The path to the root of the git repo."""
        if cls.__root_dir_attr is None:
            # This setup.py file lives in the root of the repo.
            cls.__root_dir_attr = str(Path(__file__).parent.resolve())
        return str(cls.__root_dir_attr)

    @classmethod
    def git_hash(cls) -> Optional[str]:
        """The current git hash, if known."""
        if cls.__git_hash_attr is None:
            import subprocess

            try:
                cls.__git_hash_attr = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=cls._root_dir()
                    )
                    .decode("ascii")
                    .strip()
                )
            except subprocess.CalledProcessError:
                cls.__git_hash_attr = ""  # Non-None but empty.
        # A non-None but empty value indicates that we don't know it.
        return cls.__git_hash_attr if cls.__git_hash_attr else None

    @classmethod
    def string(cls) -> str:
        """The version string."""
        if cls.__string_attr is None:
            # If set, BUILD_VERSION should override any local version
            # information. CI will use this to manage, e.g., release vs. nightly
            # versions.
            version = os.getenv("BUILD_VERSION", "").strip()
            if not version:
                # Otherwise, read the version from a local file and add the git
                # commit if available.
                version = (
                    open(os.path.join(cls._root_dir(), "version.txt")).read().strip()
                )
                if cls.git_hash():
                    version += "+" + cls.git_hash()[:7]  # type: ignore[index]
            cls.__string_attr = version
        return cls.__string_attr

    @classmethod
    def write_to_python_file(cls, path: str) -> None:
        """Creates a file similar to PyTorch core's `torch/version.py`."""

        lines = [
            "from typing import Optional",
            '__all__ = ["__version__", "git_version"]',
            f'__version__ = "{cls.string()}"',
            # A string or None.
            f"git_version: Optional[str] = {repr(cls.git_hash())}",
        ]
        with open(path, "w") as fp:
            fp.write("\n".join(lines) + "\n")


# The build type is determined by the DEBUG environment variable. If DEBUG is
# set to a non-empty value, the build type is Debug. Otherwise, the build type
# is Release.
def get_build_type(is_debug=None) -> str:
    debug = int(os.environ.get("DEBUG", 0) or 0) if is_debug is None else is_debug
    return "Debug" if debug else "Release"


def get_dynamic_lib_name(name: str) -> str:
    if _is_windows():
        return f"{name}.dll"
    elif _is_macos():
        return f"lib{name}.dylib"
    else:
        return f"lib{name}.so"


def _read_soname(lib_path: Path) -> str:
    """Return the ELF SONAME of a shared library.

    Used to recreate the loader-visible SONAME symlink for a versioned .so in
    the wheel. Raises if the SONAME cannot be read: without it the wheel would
    link at build time but fail to load at runtime (the SONAME recorded in
    dependents would have no matching file), so this must fail the build loudly
    rather than silently ship a broken wheel.
    """
    try:
        out = subprocess.run(
            ["readelf", "-d", os.fspath(lib_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            f"Could not read the SONAME of {lib_path} via readelf: {e}. "
            "The shared-library SONAME symlink cannot be created, which would "
            "produce a wheel that fails to load at runtime."
        ) from e
    for line in out.splitlines():
        if "SONAME" in line and "[" in line and "]" in line:
            return line[line.index("[") + 1 : line.index("]")]
    raise RuntimeError(
        f"{lib_path} has no ELF SONAME; cannot create the SONAME symlink for "
        "the wheel. Expected a versioned shared library (e.g. libfoo.so.1.2.3)."
    )


def get_executable_name(name: str) -> str:
    if _is_windows():
        return name + ".exe"
    else:
        return name


class _BaseExtension(Extension):
    """A base class that maps an abstract source to an abstract destination."""

    def __init__(
        self,
        src: str,
        dst: str,
        name: str,
        dependent_cmake_flags: List[str],
    ):
        # Source path; semantics defined by the subclass.
        self.src: str = src

        # Destination path relative to a namespace defined elsewhere. If this ends
        # in "/", it is treated as a directory. If this is "", it is treated as the
        # root of the namespace.
        # Destination path; semantics defined by the subclass.
        self.dst: str = dst

        # Other parts of setuptools expects .name to exist. For actual extensions
        # this can be the module path, but otherwise it should be somehing unique
        # that doesn't look like a module path.
        self.name: str = name

        self.dependent_cmake_flags = dependent_cmake_flags
        self.cmake_cache: Optional[CMakeCache] = None

        super().__init__(name=self.name, sources=[])

    def _get_build_dir(self, installer: "InstallerBuildExt") -> Path:
        # Share the cmake-out location with CustomBuild.
        build_cmd = installer.get_finalized_command("build")
        if "%CMAKE_CACHE_DIR%" in self.src:
            if not hasattr(build_cmd, "cmake_cache_dir"):
                raise RuntimeError(
                    f"Extension {self.name} has a src {self.src} that contains"
                    " %CMAKE_CACHE_DIR% but CMake does not run in the `build` "
                    "command. Please double check if the command is correct."
                )
            else:
                return Path(build_cmd.cmake_cache_dir)
        else:
            # If the src path doesn't contain %CMAKE_CACHE_DIR% placeholder,
            # try to find it under the current directory.
            return Path(".")

    def is_cmake_artifact_used(self, installer: "InstallerBuildExt") -> bool:
        cache_path = str(self._get_build_dir(installer) / "CMakeCache.txt")
        if not os.path.exists(cache_path):
            # If this is not a CMake folder, then assume it's used.
            return True
        elif self.cmake_cache is None:
            self.cmake_cache = CMakeCache(cache_path=cache_path)

        return all(
            self.cmake_cache.is_enabled(flag) for flag in self.dependent_cmake_flags
        )

    def src_path(self, installer: "InstallerBuildExt") -> Path:
        """Returns the path to the source file, resolving globs.

        Args:
            installer: The InstallerBuildExt instance that is installing the
                file.
        """
        build_dir = self._get_build_dir(installer)

        src_path = self.src.replace("%CMAKE_CACHE_DIR%/", "")

        cfg = get_build_type(installer.debug)

        if os.name == "nt":
            # Replace %BUILD_TYPE% with the current build type.
            src_path = src_path.replace("%BUILD_TYPE%", cfg)
        else:
            # Remove %BUILD_TYPE% from the path.
            src_path = src_path.replace("/%BUILD_TYPE%", "")

        # Construct the full source path, resolving globs. If there are no glob
        # pattern characters, this will just ensure that the source file exists.
        srcs = tuple(build_dir.glob(src_path))
        if len(srcs) != 1:
            raise ValueError(
                f"Expecting exactly 1 file matching {self.src} in {build_dir}, "
                f"found {repr(srcs)}. Resolved src pattern: {src_path}."
            )
        return srcs[0]

    def inplace_dir(self, installer: "InstallerBuildExt") -> Path:
        """Returns the path of this file to be installed to, under inplace mode.

        It will be a relative path to the project root directory. For more info
        related to inplace/editable mode, please checkout this doc:
        https://setuptools.pypa.io/en/latest/userguide/development_mode.html
        """
        raise NotImplementedError()


class BuiltFile(_BaseExtension):
    """An extension that installs a single file that was built by cmake.

    This isn't technically a `build_ext` style python extension, but there's no
    dedicated command for installing arbitrary data. It's convenient to use
    this, though, because it lets us manage the files to install as entries in
    `ext_modules`.
    """

    def __init__(
        self,
        src_dir: str,
        src_name: str,
        dst: str,
        dependent_cmake_flags: List[str],
        is_executable: bool = False,
        is_dynamic_lib: bool = False,
    ):
        """Initializes a BuiltFile.

        Args:
            src_dir: The directory of the file to install, relative to the cmake-out
                directory. A placeholder %BUILD_TYPE% will be replaced with the build
                type for multi-config generators (like Visual Studio) where the build
                output is in a subdirectory named after the build type. For single-
                config generators (like Makefile Generators or Ninja), this placeholder
                will be removed.
            src_name: The name of the file to install
            dst: The path to install to, relative to the root of the pip
                package. If dst ends in "/", it is treated as a directory.
                Otherwise it is treated as a filename.
            is_executable: If True, the file is an executable. This is used to
                determine the destination filename for executable.
            is_dynamic_lib: If True, the file is a dynamic library. This is used
                to determine the destination filename for dynamic library.
        """
        if is_executable and is_dynamic_lib:
            raise ValueError("is_executable and is_dynamic_lib cannot be both True.")
        if is_executable:
            src_name = get_executable_name(src_name)
        elif is_dynamic_lib:
            src_name = get_dynamic_lib_name(src_name)
        src = os.path.join(src_dir, src_name)
        # This is not a real extension, so use a unique name that doesn't look
        # like a module path. Some of setuptools's autodiscovery will look for
        # extension names with prefixes that match certain module paths.
        super().__init__(
            src=src,
            dst=dst,
            name=f"@EXECUTORCH_BuiltFile_{src}:{dst}",
            dependent_cmake_flags=dependent_cmake_flags,
        )

    def dst_path(self, installer: "InstallerBuildExt") -> Path:
        """Returns the path to the destination file.

        Args:
            installer: The InstallerBuildExt instance that is installing the
                file.
        """
        dst_root = Path(installer.build_lib).resolve()

        if self.dst.endswith("/"):
            # Destination looks like a directory. Use the basename of the source
            # file for its final component.
            return dst_root / Path(self.dst) / self.src_path(installer).name
        else:
            # Destination looks like a file.
            return dst_root / Path(self.dst)

    def inplace_dir(self, installer: "InstallerBuildExt") -> Path:
        """For a `BuiltFile`, we use self.dst as its inplace directory path.
        Need to handle directory vs file.
        """
        # HACK: get rid of the leading "executorch" in ext.dst.
        # This is because we don't have a root level "executorch" module.
        package_dir = self.dst.removeprefix("executorch/")
        # If dst is a file, use it's directory
        if not package_dir.endswith("/"):
            package_dir = os.path.dirname(package_dir)
        return Path(package_dir)


class BuiltSharedLib(BuiltFile):
    """Installs a SONAME-versioned shared library plus its symlink chain.

    A normal ``BuiltFile`` copies one file. A shared library like
    ``libexecutorch.so.1.4.0`` also needs the loader-visible SONAME symlink
    (``libexecutorch.so.1``) and the developer symlink (``libexecutorch.so``),
    or a consumer that links ``-lexecutorch`` fails at runtime because the
    SONAME recorded in dependents cannot be found. This recreates that chain in
    the wheel, matching a standard ``cmake --install`` layout.
    """

    def __init__(
        self,
        src_dir: str,
        src_name: str,
        dst: str,
        dependent_cmake_flags: Optional[List[str]] = None,
    ):
        # src_name is the base library name (e.g. "executorch"); the real file
        # is libexecutorch.so.<version>, resolved by glob in src_path().
        super().__init__(
            src_dir=src_dir,
            src_name=f"lib{src_name}.so.*",
            dst=dst,
            dependent_cmake_flags=dependent_cmake_flags or [],
        )

    def src_path(self, installer: "InstallerBuildExt") -> Path:
        # The glob matches the versioned real file and any symlinks; pick the
        # regular file (the real library), not the symlinks.
        build_dir = self._get_build_dir(installer)
        pattern = self.src.replace("%CMAKE_CACHE_DIR%/", "")
        matches = [
            p for p in build_dir.glob(pattern) if p.is_file() and not p.is_symlink()
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expecting exactly 1 real shared library matching {self.src} "
                f"in {build_dir}, found {matches}."
            )
        return matches[0]


class BuiltExtension(_BaseExtension):
    """An extension that installs a python extension that was built by cmake."""

    def __init__(
        self,
        src: str,
        modpath: str,
        dependent_cmake_flags: List[str],
        src_dir: Optional[str] = None,
    ):
        """Initializes a BuiltExtension.

        Args:
            src_dir: The directory of the file to install, relative to the cmake-out
                directory. A placeholder %BUILD_TYPE% will be replaced with the build
                type for multi-config generators (like Visual Studio) where the build
                output is in a subdirectory named after the build type. For single-
                config generators (like Makefile Generators or Ninja), this placeholder
                will be removed.
            src_name: The name of the file to install. If the path ends in `.so`,
            modpath: The dotted path of the python module that maps to the
                extension.
        """
        assert (
            "/" not in modpath
        ), f"modpath must be a dotted python module path: saw '{modpath}'"
        full_src = src
        if src_dir is None and _is_windows():
            src_dir = "%BUILD_TYPE%/"
        if src_dir is not None:
            full_src = os.path.join(src_dir, src)
        self.dependent_cmake_flags = dependent_cmake_flags
        # This is a real extension, so use the modpath as the name.
        super().__init__(
            src=f"%CMAKE_CACHE_DIR%/{full_src}",
            dst=modpath,
            name=modpath,
            dependent_cmake_flags=self.dependent_cmake_flags,
        )

    def src_path(self, installer: "InstallerBuildExt") -> Path:
        """Returns the path to the source file, resolving globs.

        Args:
            installer: The InstallerBuildExt instance that is installing the
                file.
        """
        try:
            return super().src_path(installer)
        except ValueError:
            # Probably couldn't find the file. If the path ends with .so, try
            # looking for a .dylib file instead, in case we're running on macos.
            if self.src.endswith(".so"):
                dylib_src = re.sub(r"\.so$", ".dylib", self.src)
                return BuiltExtension(
                    src=dylib_src,
                    modpath=self.dst,
                    dependent_cmake_flags=self.dependent_cmake_flags,
                ).src_path(installer)
            else:
                raise

    def dst_path(self, installer: "InstallerBuildExt") -> Path:
        """Returns the path to the destination file.

        Args:
            installer: The InstallerBuildExt instance that is installing the
                file.
        """
        # Our destination is a dotted module path. get_ext_fullpath() returns
        # the relative path to the .so/.dylib/etc. file that maps to the module
        # path: that's the file we're creating.
        return Path(installer.get_ext_fullpath(self.dst))

    def inplace_dir(self, installer: "InstallerBuildExt") -> Path:
        """For BuiltExtension, deduce inplace dir path from extension name."""
        build_py = installer.get_finalized_command("build_py")
        modpath = self.name.split(".")
        package = ".".join(modpath[:-1])
        package_dir = os.path.abspath(build_py.get_package_dir(package))

        return Path(package_dir)


class InstallerBuildExt(build_ext):
    """Installs files that were built by cmake."""

    def __init__(self, *args, **kwargs):
        self._ran_build = False
        super().__init__(*args, **kwargs)

    def run(self):
        # Run the build command first in editable mode. Since `build` command
        # will also trigger `build_ext` command, only run this once.
        if self._ran_build:
            return

        if self.editable_mode:
            self._ran_build = True
            self.run_command("build")
        super().run()

    def copy_extensions_to_source(self) -> None:
        """For each extension in `ext_modules`, we need to copy the extension
        file from the build directory to the correct location in the local
        directory.

        This should only be triggered when inplace mode (editable mode) is enabled.

        Args:

        Returns:
        """
        for ext in self.extensions:
            if not ext.is_cmake_artifact_used(self):
                continue

            package_dir = ext.inplace_dir(self)

            # Ensure that the destination directory exists.
            self.mkpath(os.fspath(package_dir))

            regular_file = ext.src_path(self)
            inplace_file = os.path.join(
                package_dir, os.path.basename(ext.dst_path(self))
            )

            # Always copy, even if source is older than destination, to ensure
            # that the right extensions for the current Python/platform are
            # used.
            if os.path.exists(regular_file) or not ext.optional:
                self.copy_file(regular_file, inplace_file, level=self.verbose)

            if ext._needs_stub:
                inplace_stub = self._get_equivalent_stub(ext, inplace_file)
                self._write_stub_file(inplace_stub, ext, compile=True)
                # Always compile stub and remove the original (leave the cache behind)
                # (this behaviour was observed in previous iterations of the code)

    # TODO(dbort): Depend on the "build" command to ensure it runs first

    def build_extension(self, ext: _BaseExtension) -> None:
        if not ext.is_cmake_artifact_used(self):
            return

        src_file: Path = ext.src_path(self)
        dst_file: Path = ext.dst_path(self)

        # Ensure that the destination directory exists.
        if not dst_file.parent.exists():
            self.mkpath(os.fspath(dst_file.parent))

        # Copy the file.
        self.copy_file(os.fspath(src_file), os.fspath(dst_file))

        # For a SONAME-versioned shared library, normalize what ships in the
        # wheel. pip does not preserve symlinks (it materializes every wheel
        # entry as a regular file), so a libexecutorch.so -> .so.1 -> .so.1.2.3
        # chain would become three identical multi-hundred-KB copies. Instead
        # ship exactly two files: the SONAME (libexecutorch.so.1, what the loader
        # resolves via DT_NEEDED) as the real library, and the dev name
        # (libexecutorch.so, what -lexecutorch / find_library(NAMES executorch)
        # need). On a filesystem the dev name is a relative symlink; if the wheel
        # packer dereferences it, it becomes at most one extra copy, not two.
        if isinstance(ext, BuiltSharedLib):
            self._normalize_shared_lib(src_file, dst_file)

    def _normalize_shared_lib(self, src_file: Path, dst_file: Path) -> None:
        lib_dir = dst_file.parent
        # SONAME (e.g. libexecutorch.so.1) read from the built library. Required,
        # not best-effort: it is what the loader resolves at runtime.
        # _read_soname raises if it cannot be determined.
        soname = _read_soname(src_file)
        soname_path = lib_dir / soname
        # Make the SONAME file the real library. dst_file is the fully versioned
        # name (e.g. libexecutorch.so.1.4.0) that copy_file just wrote; rename it
        # to the SONAME so we do not keep a third redundant copy.
        if dst_file.name != soname:
            if soname_path.exists() or soname_path.is_symlink():
                soname_path.unlink()
            os.replace(os.fspath(dst_file), os.fspath(soname_path))

        # Dev name (drop all version suffixes), e.g. libexecutorch.so.
        dev_name = soname.split(".so")[0] + ".so"
        if dev_name != soname:
            dev_path = lib_dir / dev_name
            if dev_path.exists() or dev_path.is_symlink():
                dev_path.unlink()
            # Relative link so the wheel is relocatable on a real filesystem.
            os.symlink(soname, os.fspath(dev_path))

        # Ensure the shipped library is writable (the CMake output may be
        # read-only), so a later rebuild/repack can overwrite it. Mirrors the
        # writability fix build_py applies via copy_file(preserve_mode=False),
        # but scoped to this file so it does not clobber X bits elsewhere.
        if not os.access(soname_path, os.W_OK):
            os.chmod(soname_path, os.stat(soname_path).st_mode | 0o222)


class CustomBuildPy(build_py):
    """Copies platform-independent files from the source tree into the output
    package directory.

    Override it so we can copy some files to locations that don't match their
    original relative locations.

    Standard setuptools features like package_data and MANIFEST.in can only
    include or exclude a file in the source tree; they don't have a way to map
    a file to a different relative location under the output package directory.
    """

    def run(self):
        # Copy python files to the output directory. This set of files is
        # defined by the py_module list and package_data patterns.
        build_py.run(self)

        # dst_root is the root of the `executorch` module in the output package
        # directory. build_lib is the platform-independent root of the output
        # package, and will look like `pip-out/lib`. It can contain multiple
        # python packages, so be sure to copy the files into the `executorch`
        # package subdirectory.
        if self.editable_mode:
            # In editable mode, the package directory is the original source directory
            dst_root = self.get_package_dir("executorch")
        else:
            dst_root = os.path.join(self.build_lib, "executorch")
        # Create the version file.
        Version.write_to_python_file(os.path.join(dst_root, "version.py"))

        # Manually copy files into the output package directory. These are
        # typically python "resource" files that will live alongside the python
        # code that uses them.
        src_to_dst = [
            # TODO(dbort): See if we can add a custom pyproject.toml section for
            # these, instead of hard-coding them here. See
            # https://setuptools.pypa.io/en/latest/userguide/extension.html
            ("schema/scalar_type.fbs", "exir/_serialize/scalar_type.fbs"),
            ("schema/program.fbs", "exir/_serialize/program.fbs"),
        ]
        if not _is_minimal_build():
            src_to_dst += [
                (
                    "devtools/bundled_program/schema/bundled_program_schema.fbs",
                    "devtools/bundled_program/serialize/bundled_program_schema.fbs",
                ),
                (
                    "devtools/bundled_program/schema/scalar_type.fbs",
                    "devtools/bundled_program/serialize/scalar_type.fbs",
                ),
                # Install executorch-wheel-config.cmake to pip package.
                (
                    "tools/cmake/executorch-wheel-config.cmake",
                    "share/cmake/executorch-config.cmake",
                ),
            ]
            # Copy all the necessary headers into include/executorch/ so that they can
            # be found in the pip package. This is the subset of headers that are
            # essential for building custom ops extensions.
            # TODO: Use cmake to gather the headers instead of hard-coding them here.
            # For example:
            # https://discourse.cmake.org/t/installing-headers-the-modern-way-regurgitated-and-revisited/3238/3
            for include_dir in [
                "runtime/core/",
                "runtime/executor/",
                "runtime/kernel/",
                "runtime/backend/",
                "runtime/platform/",
                "extension/kernel_util/",
                "extension/tensor/",
                "extension/threadpool/",
            ]:
                src_list = Path(include_dir).rglob("*.h")
                for src in src_list:
                    src_to_dst.append(
                        (str(src), os.path.join("include/executorch", str(src)))
                    )
            # Delegate-only C++ SDK headers: the Program/Module/Tensor/DataLoader/
            # .ptd APIs a standalone C++ runner needs, so it can link the prebuilt
            # runtime without an ExecuTorch source tree. These are listed
            # explicitly (not rglob) so we only advertise APIs whose implementation
            # archive is actually shipped in executorch/lib/. Excluded on purpose:
            # bundled_module.h (needs the separate bundled_module archive),
            # flat_tensor/serialize/serialize.h (serializer .cpp not shipped),
            # file_descriptor_data_loader.h (impl not in the shipped archive), and
            # cpu_caching_malloc_allocator.h (needs a memory_allocator archive we
            # do not ship). flat_tensor_header.h IS shipped: it is the .ptd reader.
            # Linux only, matching the SDK archives below; keeps Windows/macOS
            # wheels unchanged.
            sdk_headers = (
                [
                    "extension/module/module.h",
                    "extension/data_loader/buffer_data_loader.h",
                    "extension/data_loader/file_data_loader.h",
                    "extension/data_loader/mmap_data_loader.h",
                    "extension/data_loader/mman.h",
                    "extension/data_loader/mman_windows.h",
                    "extension/data_loader/shared_ptr_data_loader.h",
                    "extension/flat_tensor/flat_tensor_data_map.h",
                    "extension/flat_tensor/serialize/flat_tensor_header.h",
                    "extension/named_data_map/merged_data_map.h",
                    "extension/memory_allocator/malloc_memory_allocator.h",
                    "extension/memory_allocator/memory_allocator_utils.h",
                    # CUDA caller-stream extension headers. Harmless on a CPU
                    # wheel (headers only); the matching libextension_cuda.so and
                    # the executorch::extension_cuda / executorch::cuda_backend
                    # CMake targets are shipped/defined only in a CUDA wheel.
                    "extension/cuda/caller_stream.h",
                    "extension/cuda/export.h",
                ]
                if sys.platform == "linux"
                else []
            )
            for src in sdk_headers:
                src_to_dst.append((src, os.path.join("include/executorch", src)))
        for src, dst in src_to_dst:
            dst = os.path.join(dst_root, dst)

            # When modifying the filesystem, use the self.* methods defined by
            # Command to benefit from the same logging and dry_run logic as
            # setuptools.

            # Ensure that the destination directory exists.
            self.mkpath(os.path.dirname(dst))
            # Follow the example of the base build_py class by not preserving
            # the mode. This ensures that the output file is read/write even if
            # the input file is read-only.
            self.copy_file(src, dst, preserve_mode=False)

        # Copy CMake-generated Python directories that setuptools missed.
        # Setuptools discovers packages at configuration time, before CMake
        # runs. Directories created by CMake during the build (e.g. by
        # generate.py) are not in the package list and must be copied manually.
        generated_dirs = []
        if not _is_minimal_build():
            generated_dirs.append("backends/mlx/serialization/_generated")
        for rel_dir in generated_dirs:
            src_dir = os.path.join("src/executorch", rel_dir)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(dst_root, rel_dir)
            for dirpath, _dirnames, filenames in os.walk(src_dir):
                for filename in filenames:
                    src_file = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(src_file, src_dir)
                    dst_file = os.path.join(dst_dir, rel_path)
                    self.mkpath(os.path.dirname(dst_file))
                    self.copy_file(src_file, dst_file, preserve_mode=False)


class Buck2EnvironmentFixer(contextlib.AbstractContextManager):
    """Removes HOME from the environment when running as root.

    This script is sometimes run as root in docker containers. buck2 doesn't
    allow running as root unless $HOME is owned by root or is not set.

    TODO(pytorch/test-infra#5091): Remove this once the CI jobs stop running as
    root.
    """

    def __init__(self):
        self.saved_env = {}

    def __enter__(self):
        if os.name != "nt" and os.geteuid() == 0 and "HOME" in os.environ:
            log.info("temporarily unsetting HOME while running as root")
            self.saved_env["HOME"] = os.environ.pop("HOME")
        return self

    def __exit__(self, *args, **kwargs):
        if "HOME" in self.saved_env:
            log.info("restored HOME")
            os.environ["HOME"] = self.saved_env["HOME"]


# TODO(dbort): For editable wheels, may need to update get_source_files(),
# get_outputs(), and get_output_mapping() to satisfy
# https://setuptools.pypa.io/en/latest/userguide/extension.html#setuptools.command.build.SubCommand.get_output_mapping


class CustomBuild(build):
    def initialize_options(self):
        super().initialize_options()
        # The default build_base directory is called "build", but we have a
        # top-level directory with that name. Setting build_base in setup()
        # doesn't affect this, so override the core build command.
        #
        # See build.initialize_options() in
        # setuptools/_distutils/command/build.py for the default.
        self.build_base = "pip-out"

    def run(self):  # noqa C901
        self.dump_options()
        minimal_build = _is_minimal_build()
        cmake_build_type = get_build_type(self.debug)
        # get_python_lib() typically returns the path to site-packages, where
        # all pip packages in the environment are installed.
        cmake_prefix_path = os.environ.get("CMAKE_PREFIX_PATH", get_python_lib())
        # Put the cmake cache under the temp directory, like
        # "pip-out/temp.<plat>/cmake-out".
        pip_build_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), self.build_temp
        )
        cmake_cache_dir = os.path.join(pip_build_dir, "cmake-out")
        self.mkpath(cmake_cache_dir)

        cmake_configuration_args = [
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            # Let cmake calls like `find_package(Torch)` find cmake config files
            # like `TorchConfig.cmake` that are provided by pip packages.
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DCMAKE_BUILD_TYPE={cmake_build_type}",
        ]

        # Use ClangCL on Windows.
        if _is_windows():
            cmake_configuration_args += ["-T ClangCL"]

        # Allow adding extra cmake args through the environment. Used by some
        # tests and demos to expand the set of targets included in the pip
        # package.
        cmake_configuration_args += [
            item for item in re.split(r"\s+", os.environ.get("CMAKE_ARGS", "")) if item
        ]
        if minimal_build:
            cmake_configuration_args += _minimal_cmake_flags()

        # Check if CUDA is available, and if so, enable building the CUDA
        # backend by default.
        if (
            not minimal_build
            and install_utils.is_cuda_available()
            and install_utils.is_cmake_option_on(
                cmake_configuration_args, "EXECUTORCH_BUILD_CUDA", default=True
            )
        ):
            cmake_configuration_args += ["-DEXECUTORCH_BUILD_CUDA=ON"]

        # Check if QNN SDK is available (via QNN_SDK_ROOT env var), and if so,
        # enable building the Qualcomm backend by default.
        qnn_sdk_root = os.environ.get("QNN_SDK_ROOT", "").strip()
        if (
            not minimal_build
            and qnn_sdk_root
            and install_utils.is_cmake_option_on(
                cmake_configuration_args, "EXECUTORCH_BUILD_QNN", default=True
            )
        ):
            cmake_configuration_args += [
                "-DEXECUTORCH_BUILD_QNN=ON",
                f"-DQNN_SDK_ROOT={qnn_sdk_root}",
            ]

        # Enable OpenVINO backend on Linux. The backend uses dlopen at
        # runtime so it has no build-time SDK dependency.
        if (
            not minimal_build
            and sys.platform == "linux"
            and install_utils.is_cmake_option_on(
                cmake_configuration_args,
                "EXECUTORCH_BUILD_OPENVINO",
                default=True,
            )
        ):
            cmake_configuration_args += ["-DEXECUTORCH_BUILD_OPENVINO=ON"]

        # Build the consolidated shared runtime libexecutorch.so so the wheel can
        # ship a linkable C++ SDK (see the executorch_shared build target and the
        # packaging step). Only for an actual wheel build: EXECUTORCH_BUILD_SHARED
        # also forces global PIC and changes link behavior, so it must not leak
        # into editable/develop/install builds that other CI (unittest, arm, etc.)
        # rely on. Linux only; the SDK is not shipped on Windows/macOS.
        building_wheel = "bdist_wheel" in self.distribution.commands
        if not minimal_build and sys.platform == "linux" and building_wheel:
            cmake_configuration_args += ["-DEXECUTORCH_BUILD_SHARED=ON"]

        with Buck2EnvironmentFixer():
            # Generate the cmake cache from scratch to ensure that the cache state
            # is predictable.
            if os.path.exists(cmake_cache_dir):
                log.info(f"clearing {cmake_cache_dir}")
                shutil.rmtree(cmake_cache_dir)

            subprocess.run(
                [
                    "cmake",
                    *cmake_configuration_args,
                    "--preset",
                    "pybind",
                    "-B",
                    cmake_cache_dir,
                ],
                check=True,
            )

        cmake_cache = CMakeCache(
            cache_path=os.path.join(cmake_cache_dir, "CMakeCache.txt")
        )
        cmake_build_args = [
            # Default build parallelism based on number of cores, but allow
            # overriding through the environment.
            "-j{parallelism}".format(
                parallelism=os.environ.get(
                    "CMAKE_BUILD_PARALLEL_LEVEL", os.cpu_count() - 1
                )
            ),
            # CMAKE_BUILD_TYPE variable specifies the build type (configuration) for
            # single-configuration generators (e.g., Makefile Generators or Ninja).
            # For multi-config generators (like Visual Studio), CMAKE_BUILD_TYPE
            # isn’t directly applicable.
            # During the build step, --config specifies the configuration to build
            # for multi-config generators.
            f"--config={cmake_build_type}",
        ]

        # Allow adding extra build args through the environment. Used by some
        # tests and demos to expand the set of targets included in the pip
        # package.
        cmake_build_args += [
            item
            for item in re.split(r"\s+", os.environ.get("CMAKE_BUILD_ARGS", ""))
            if item
        ]

        if minimal_build:
            # The minimal wheel only needs flatc. Every other target is gated off
            # by _minimal_cmake_flags(), so skip the entire non-minimal target
            # list explicitly rather than relying on each flag being OFF.
            cmake_build_args += ["--target", "flatbuffers_ep"]
        else:
            # Delegate-only C++ SDK: ship the consolidated shared runtime
            # libexecutorch.so (the executorch_shared target), which bundles the
            # runtime core plus the common extensions (module, tensor,
            # data_loader, flat_tensor, named_data_map). Shared is required so a
            # separately distributed backend/delegate .so can register into the
            # one process-global registry inside libexecutorch.so. Built only for
            # a wheel build (guarded by the EXECUTORCH_BUILD_SHARED cache entry,
            # which is set just above only for bdist_wheel), so editable/develop
            # builds are unaffected and the target always exists when requested.
            if sys.platform == "linux" and cmake_cache.is_enabled(
                "EXECUTORCH_BUILD_SHARED"
            ):
                cmake_build_args += ["--target", "executorch_shared"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_PYBIND"):
                cmake_build_args += ["--target", "portable_lib"]
                cmake_build_args += ["--target", "data_loader"]
                cmake_build_args += ["--target", "selective_build"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER"):
                cmake_build_args += ["--target", "_llm_runner"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_CUDA"):
                cmake_build_args += ["--target", "aoti_cuda_backend"]
                cmake_build_args += ["--target", "aoti_common_shims_slim"]
                # Loadable CUDA delegate + caller-stream shared libs for the C++
                # SDK, built only for a wheel that also builds the shared runtime.
                if cmake_cache.is_enabled("EXECUTORCH_BUILD_SHARED"):
                    cmake_build_args += ["--target", "executorch_cuda_backend"]
                    cmake_build_args += ["--target", "extension_cuda"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_EXTENSION_MODULE"):
                cmake_build_args += ["--target", "extension_module"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_EXTENSION_TRAINING"):
                cmake_build_args += ["--target", "_training_lib"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_COREML"):
                cmake_build_args += ["--target", "executorchcoreml"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_MLX"):
                cmake_build_args += ["--target", "mlxdelegate"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_KERNELS_LLM_AOT"):
                cmake_build_args += ["--target", "custom_ops_aot_lib"]
                cmake_build_args += ["--target", "quantized_ops_aot_lib"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_QNN"):
                cmake_build_args += ["--target", "qnn_executorch_backend"]
                cmake_build_args += ["--target", "PyQnnManagerAdaptor"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_OPENVINO"):
                cmake_build_args += ["--target", "openvino_backend"]

        # Set PYTHONPATH to the location of the pip package.
        os.environ["PYTHONPATH"] = (
            site.getsitepackages()[0] + ";" + os.environ.get("PYTHONPATH", "")
        )
        # Build the system.
        self.spawn(["cmake", "--build", cmake_cache_dir, *cmake_build_args])
        # Share the cmake-out location with _BaseExtension.
        self.cmake_cache_dir = cmake_cache_dir
        # Finally, run the underlying subcommands like build_py, build_ext.
        build.run(self)


setup_kwargs = {}
if _is_minimal_build():
    setup_kwargs["packages"] = _minimal_packages()
    setup_kwargs["install_requires"] = _minimal_dependencies()
else:
    setup_kwargs["install_requires"] = _base_dependencies()


setup(
    version=Version.string(),
    cmdclass={
        "build": CustomBuild,
        "build_ext": InstallerBuildExt,
        "build_py": CustomBuildPy,
    },
    # Note that setuptools uses the presence of ext_modules as the main signal
    # that a wheel is platform-specific. If we install any platform-specific
    # files, this list must be non-empty. Therefore, we should always install
    # platform-specific files using InstallerBuildExt.
    ext_modules=[
        BuiltFile(
            src_dir="%CMAKE_CACHE_DIR%/third-party/flatc_ep/bin/",
            src_name="flatc",
            dst="executorch/data/bin/",
            is_executable=True,
            dependent_cmake_flags=[],
        ),
        BuiltFile(
            src_dir="tools/wheel",
            src_name="pip_data_bin_init.py.in",
            dst="executorch/data/bin/__init__.py",
            dependent_cmake_flags=[],
        ),
        *(
            []
            if _is_minimal_build()
            else [
                # Install the prebuilt pybindings extension wrapper for the runtime,
                # portable kernels, and a selection of backends. This lets users
                # load and execute .pte files from python.
                BuiltExtension(
                    src="_portable_lib.cp*" if _is_windows() else "_portable_lib.*",
                    modpath="executorch.extension.pybindings._portable_lib",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_PYBIND"],
                ),
                # Install the data_loader pybindings extension which provides the
                # PyDataLoader type for external pybinding extensions.
                BuiltExtension(
                    src="data_loader.cp*" if _is_windows() else "data_loader.*",
                    modpath="executorch.extension.pybindings.data_loader",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_PYBIND"],
                ),
                # MLX metallib (Metal GPU kernels) must be colocated with _portable_lib.so
                # because MLX uses dladdr() to find the directory containing the library,
                # then looks for mlx.metallib in that directory at runtime.
                # After submodule migration, the path is backends/mlx/mlx/...
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/backends/mlx/mlx/mlx/backend/metal/kernels/",
                    src_name="mlx.metallib",
                    dst="executorch/extension/pybindings/",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_MLX"],
                ),
                BuiltExtension(
                    src="extension/training/_training_lib.*",  # @lint-ignore https://github.com/pytorch/executorch/blob/cb3eba0d7f630bc8cec0a9cc1df8ae2f17af3f7a/scripts/lint_xrefs.sh
                    modpath="executorch.extension.training.pybindings._training_lib",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_EXTENSION_TRAINING"],
                ),
                BuiltExtension(
                    src_dir="%CMAKE_CACHE_DIR%/codegen/tools/%BUILD_TYPE%/",
                    src="selective_build.cp*" if _is_windows() else "selective_build.*",
                    modpath="executorch.codegen.tools.selective_build",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_PYBIND"],
                ),
                BuiltExtension(
                    src="extension/llm/runner/_llm_runner.*",  # @lint-ignore https://github.com/pytorch/executorch/blob/cb3eba0d7f630bc8cec0a9cc1df8ae2f17af3f7a/scripts/lint_xrefs.sh
                    modpath="executorch.extension.llm.runner._llm_runner",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER"],
                ),
                BuiltExtension(
                    src="executorchcoreml.*",
                    src_dir="backends/apple/coreml",
                    modpath="executorch.backends.apple.coreml.executorchcoreml",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_COREML"],
                ),
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/extension/llm/custom_ops/%BUILD_TYPE%/",
                    src_name="custom_ops_aot_lib",
                    dst="executorch/extension/llm/custom_ops/",
                    is_dynamic_lib=True,
                    dependent_cmake_flags=["EXECUTORCH_BUILD_KERNELS_LLM_AOT"],
                ),
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/kernels/quantized/%BUILD_TYPE%/",
                    src_name="quantized_ops_aot_lib",
                    dst="executorch/kernels/quantized/",
                    is_dynamic_lib=True,
                    dependent_cmake_flags=["EXECUTORCH_BUILD_KERNELS_LLM_AOT"],
                ),
                BuiltFile(
                    src_dir="backends/cuda/runtime/",
                    src_name="aoti_cuda_shims.lib",
                    dst="executorch/data/lib/",
                    dependent_cmake_flags=[],
                ),
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/backends/cuda/%BUILD_TYPE%/",
                    src_name="aoti_cuda_shims",
                    dst="executorch/backends/cuda/",
                    is_dynamic_lib=True,
                    dependent_cmake_flags=["EXECUTORCH_BUILD_CUDA"],
                ),
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/backends/qualcomm/%BUILD_TYPE%/",
                    src_name="qnn_executorch_backend",
                    dst="executorch/backends/qualcomm/",
                    is_dynamic_lib=True,
                    dependent_cmake_flags=["EXECUTORCH_BUILD_QNN"],
                ),
                BuiltExtension(
                    src_dir="%CMAKE_CACHE_DIR%/backends/qualcomm/%BUILD_TYPE%/",
                    src="PyQnnManagerAdaptor.*",
                    modpath="executorch.backends.qualcomm.python.PyQnnManagerAdaptor",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_QNN"],
                ),
                # Delegate-only C++ SDK: the consolidated shared runtime
                # libexecutorch.so (core + module/tensor/data_loader/flat_tensor/
                # named_data_map), plus its SONAME symlink chain. Shipping it
                # shared (not static archives) lets a separately distributed
                # backend/delegate .so register into the one process-global
                # registry inside libexecutorch.so, which is what coalesced
                # multi-backend .pte execution requires. Paired with
                # executorch-config.cmake (executorch::runtime target) and
                # executorch.utils.cmake_prefix_path. Linux only: the .so naming
                # and symlink chain are Unix specific, so Windows/macOS wheels are
                # unchanged.
                *(
                    [
                        BuiltSharedLib(
                            src_dir="%CMAKE_CACHE_DIR%/",
                            src_name="executorch",
                            dst="executorch/lib/",
                            dependent_cmake_flags=["EXECUTORCH_BUILD_SHARED"],
                        )
                    ]
                    if sys.platform == "linux"
                    else []
                ),
                # CUDA delegate binaries for the C++ SDK, shipped only in a
                # CUDA-enabled wheel (PyTorch-style: the default wheel has the core
                # runtime, the CUDA-index wheel adds these). Co-located with
                # libexecutorch.so in executorch/lib/ so a C++ consumer, or a
                # coalesced TensorRT+CUDA .pte, can load and register the CUDA
                # delegate. executorch_cuda_backend whole-archives the backend so
                # its "CudaBackend" registration runs on load; extension_cuda
                # carries the single process-wide caller-stream TLS. Both are
                # unversioned .so, so a plain dynamic-lib copy is enough. Gated on
                # BUILD_SHARED as well as BUILD_CUDA: executorch_cuda_backend links
                # the shared runtime and is only defined/built when SHARED is on,
                # so packaging must require the same to avoid referencing an
                # unbuilt artifact.
                *(
                    [
                        BuiltFile(
                            src_dir="%CMAKE_CACHE_DIR%/backends/cuda/",
                            src_name="executorch_cuda_backend",
                            dst="executorch/lib/",
                            is_dynamic_lib=True,
                            dependent_cmake_flags=[
                                "EXECUTORCH_BUILD_CUDA",
                                "EXECUTORCH_BUILD_SHARED",
                            ],
                        ),
                        BuiltFile(
                            src_dir="%CMAKE_CACHE_DIR%/extension/cuda/",
                            src_name="extension_cuda",
                            dst="executorch/lib/",
                            is_dynamic_lib=True,
                            dependent_cmake_flags=[
                                "EXECUTORCH_BUILD_CUDA",
                                "EXECUTORCH_BUILD_SHARED",
                            ],
                        ),
                    ]
                    if sys.platform == "linux"
                    else []
                ),
            ]
        ),
    ],
    **setup_kwargs,
)
