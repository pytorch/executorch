# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024, 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Part of this code is from pybind11 cmake_example, so attach the license below.
# That project has since dropped setup.py, so this points at the last revision
# that still had it instead of at a branch.
# https://github.com/pybind/cmake_example/blob/7a94877f581a14de4de1a096fb053a55fc2a66bf/setup.py

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
import stat
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

# Headers swept in by a directory copy that a consumer of the wheel cannot use, because each needs
# something the wheel does not carry. Publishing one is worse than leaving it out: the failure arrives in
# someone else's project rather than here.
#
# Matched on the path ending, not the bare file name. Two different headers here share the name
# tensor_util.h, one a widely included utility and one a test helper, so a name match either kept the test
# helper or removed the utility everything needs.
#
# Only headers that nothing else the wheel installs includes belong here. A header other shipped headers
# pull in must keep shipping even when it cannot be compiled on its own.
_UNSHIPPABLE_HEADERS = frozenset(
    {
        # Needs a header generated when the schema is compiled, which in turn needs the FlatBuffers C++
        # headers. Those are a third-party library this wheel does not vendor.
        "runtime/executor/tensor_parser.h",
        # A test helper, needing a test framework the wheel does not ship.
        "runtime/core/testing_util/error_matchers.h",
        # Reads processor details through cpuinfo, whose headers the wheel does not publish.
        "extension/threadpool/cpuinfo_utils.h",
        # Holds a pthreadpool member by value, so it needs that library's header, which the wheel does not
        # publish either. The component it belongs to is a link dependency the runtime carries, not
        # something a consumer includes.
        "extension/threadpool/threadpool.h",
        # Declares CPUCachingAllocator, whose implementation is in a component no shipped library links,
        # so including it compiles and then fails at link time with an undefined reference.
        "extension/memory_allocator/cpu_caching_malloc_allocator.h",
        # Declares BundledModule, which is built only for the Python bindings, so its implementation is in
        # the Python extension. A C++ application cannot link that, and building the source instead needs
        # bundled-program headers the wheel does not publish.
        "extension/module/bundled_module.h",
        # Declares FileDescriptorDataLoader, whose implementation is in no CMake target at all, so no
        # shipped library defines it. Including it compiles and then fails at link time.
        "extension/data_loader/file_descriptor_data_loader.h",
    }
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
        "-DEXECUTORCH_BUILD_VULKAN=OFF",
        "-DEXECUTORCH_BUILD_XNNPACK=OFF",
        "-DEXECUTORCH_BUILD_CMSIS_NN_PYBINDS=OFF",
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
        # backends/qualcomm/__init__.py cannot be imported from a clean install
        # without both of these. It reads the CPU vendor to disable an mkldnn path on
        # AMD, and the module it imports first does a module-scope `import requests`,
        # so declaring only the cpuinfo half leaves the import failing on the line
        # before.
        "py-cpuinfo",
        "requests",
        "pytorch-tokenizers",
        "pyyaml",
        "ruamel.yaml",
        "sympy",
        "tabulate",
        # See also third-party/TARGETS for buck's typing-extensions version.
        "typing-extensions>=4.10.0",
        # Keep this version in sync with: ./backends/apple/coreml/scripts/install_requirements.sh
        "coremltools==9.0; (platform_system == 'Darwin' or platform_system == 'Linux') and python_version < '3.14'",
        # scikit-learn is used to support palettization in the coreml backend.
        "scikit-learn>=1.7.1",
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
        # The destination is relative to the installed package, so resolve it against the same
        # package directory an extension uses. Anchoring at the repo root instead only worked for
        # destinations that already had a directory under src/executorch, and silently scattered
        # the rest, which left the CMake package searching a directory nothing was copied into.
        relative = self.dst.removeprefix("executorch/")
        if not relative.endswith("/"):
            relative = os.path.dirname(relative)
        build_py = installer.get_finalized_command("build_py")
        package_dir = os.path.abspath(build_py.get_package_dir("executorch"))
        return Path(package_dir) / relative


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
        if self.editable_mode:
            # The first substitution runs before any cache exists, so it reads the command line
            # and falls back to off. The preset this build uses turns the tracer on without
            # passing anything, so that fallback published the wrong struct layout for an
            # ordinary editable install. The build has produced a cache by now, so replace the
            # best effort value with what the build actually configured.
            self._resubstitute_tracer_definition()

    def _resubstitute_tracer_definition(self) -> None:
        cache_dir = _tracer_cache_dir(self)
        if cache_dir is None:
            return
        for destination in _TRACER_DEFINITION_PATHS:
            if os.path.exists(destination):
                _substitute_tracer_definition(destination, cache_dir)

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
                # A copied extension still names its libraries by soname, and the entries that
                # reach them are relative to where it was built. The wheel path repairs that
                # from build_extension, and the editable copy needs the same repair or the
                # import fails on a library the loader cannot find.
                if isinstance(ext, BuiltExtension):
                    _strip_absolute_runtime_paths(Path(inplace_file))

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

        # Ensure that the destination file is writable, even if the source was
        # not. build_py does this by passing preserve_mode=False to copy_file,
        # but that would clobber the X bit on any executables. TODO(dbort): This
        # probably won't work on Windows.
        if not os.access(src_file, os.W_OK):
            # The owner only. A mode of 0o222 would also grant write to the group
            # and to everyone, and chmod takes an absolute mode so no umask
            # narrows it, which turned a 0o555 build output into 0o777.
            os.chmod(src_file, os.stat(src_file).st_mode | stat.S_IWUSR)

        # The destination too, and before the rewrite below, which opens the file
        # for writing. copy_file preserves mode here on purpose, because this path
        # also copies flatc and preserve_mode=False would drop its executable bit,
        # so a read-only source arrives read-only.
        #
        # This mode is the one the wheel archives, so widening it here ships a
        # world-writable library.
        if not os.access(dst_file, os.W_OK):
            os.chmod(dst_file, os.stat(dst_file).st_mode | stat.S_IWUSR)

        _strip_absolute_runtime_paths(dst_file)


def _origin_token() -> str:
    """The loader's own-directory token, spelled differently by the two formats."""
    return "@loader_path" if sys.platform == "darwin" else "$ORIGIN"


def _strip_absolute_runtime_paths(library: Path) -> None:
    """Remove unusable runtime search paths from a library the wheel ships.

    These libraries are copied out of the build tree rather than installed, so they
    still carry every directory the linker recorded while resolving their
    dependencies. Two kinds of entry are removed:

    - a directory inside this build, which names the machine that produced the
      wheel and cannot exist for a user
    - an empty entry, which the loader reads as the process working directory

    Other absolute entries are kept. The Python extensions link torch and resolve it
    through the directory the linker recorded, so dropping that would stop them
    importing in an environment where torch is not beside them.

    Best effort, because the tool cannot be guaranteed on PATH: the pip package does not
    reliably provide a binary inside a build venv, so failing the build here would break
    building from source on a machine that simply lacks it. It is declared as a build
    requirement so a release build gets it.

    A CPU wheel built without it still works, with the absolute paths left in place. A CUDA
    wheel does not: the delegate under backends/cuda/ reaches its dependency in lib/ only
    through the paths written here, so without the rewrite that edge is missing.

    What must not happen is both this and its check going quiet together, which is how
    a wheel carrying build-machine directories could ship unnoticed. So the check in
    the release tests treats a missing patchelf as a failure rather than a skip: the
    wheel-build environment has it, and that is where the guarantee belongs.
    """
    if library.suffix != ".so" and ".so." not in library.name:
        return
    patchelf = shutil.which("patchelf")
    if patchelf is None:
        return
    result = subprocess.run(
        [patchelf, "--print-rpath", os.fspath(library)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return
    original = result.stdout.strip()
    if not original:
        # No runtime search path at all, which is nothing to clean. patchelf prints
        # the same empty string for an absent tag and for one holding a single empty
        # entry, so there is nothing to distinguish here and nothing to do either way.
        return

    def keep(entry: str) -> bool:
        if not entry:
            # The loader reads an empty entry as the process working directory.
            return False
        if not entry.startswith("/"):
            return True
        # Absolute, so decide by what it points at. A directory inside this build
        # cannot exist for a user.
        #
        # A torch lib directory is dropped when a relative route to torch is already
        # recorded, since that route is what resolves torch on an installed wheel
        # while the absolute one only names a directory from the machine that built
        # it. It is kept when no relative route exists, because then it is the only
        # way this library finds torch.
        #
        # Anything else absolute is a dependency the environment provides and the
        # wheel has no relative answer for.
        #
        # Matched as whole path components rather than as substrings. A bare
        # "/cmake-out" also matches "/home/user/cmake-outputs/torchlibs", which is
        # an unrelated directory a user could really have, and stripping it breaks
        # a dependency the library legitimately resolves there.
        if entry.rstrip("/").endswith("/torch/lib") and has_relative_torch_route:
            return False
        parts = entry.split("/")
        # The setuptools staging directory is spelled build/lib.<platform>-<pyver>,
        # for example lib.linux-x86_64-cpython-312. A bare startswith("lib.") also
        # stripped a real user path like /opt/acme/lib.v2, so match the whole shape.
        return not any(
            part == "pip-out"
            or part == "cmake-out"
            or re.fullmatch(r"lib\.[^/]+-(cpython-\d+|\d+(?:\.\d+)*)", part)
            for part in parts
        )

    # Whether the library can still reach torch without the absolute entry. Read
    # inside keep, which closes over this scope.
    has_relative_torch_route = any(
        not entry.startswith("/") and entry.rstrip("/").endswith("/torch/lib")
        for entry in original.split(":")
    )
    # An editable install copies this extension into a package directory whose
    # subdirectories are symlinks to the checkout, and the loader resolves the origin
    # token against the real path, so the runtime sits inside src/executorch/lib rather
    # than beside the copy. Nothing else adds a relative route at this point.
    hop = _origin_token() + "/../../src/executorch/lib"
    rewritten = ":".join(entry for entry in original.split(":") if keep(entry))
    if hop not in rewritten.split(":"):
        rewritten = rewritten + ":" + hop if rewritten else hop
    if rewritten == original:
        return
    subprocess.run(
        [patchelf, "--set-rpath", rewritten, os.fspath(library)],
        check=True,
    )


class CustomBuildPy(build_py):
    """Copies platform-independent files from the source tree into the output
    package directory.

    Override it so we can copy some files to locations that don't match their
    original relative locations.

    Standard setuptools features like package_data and MANIFEST.in can only
    include or exclude a file in the source tree; they don't have a way to map
    a file to a different relative location under the output package directory.
    """

    def analyze_manifest(self):
        super().analyze_manifest()
        # Recent versions of setuptools may include bare directory symlinks from version
        # control (e.g. src/executorch/{backends,codegen,data,...} ->
        # ../../<name>) in manifest_files. These exist for editable mode but
        # break regular installs: build_package_data passes them to copy_file,
        # which calls os.path.isfile() and gets False for a symlink-to-directory.
        if not self.editable_mode:
            _root = os.path.dirname(os.path.abspath(__file__))
            for _pkg in list(self.manifest_files):
                self.manifest_files[_pkg] = [
                    _f
                    for _f in self.manifest_files[_pkg]
                    if os.path.isfile(os.path.join(_root, _f))
                ]

    def _copy_extra_files(self, src_to_dst, dst_root: str) -> None:
        """Copy the non-Python files the package ships, filling in any placeholders."""
        for src, dst in src_to_dst:
            dst = os.path.join(dst_root, dst)

            # When modifying the filesystem, use the self.* methods defined by
            # Command to benefit from the same logging and dry_run logic as
            # setuptools.

            # Ensure that the destination directory exists.
            self.mkpath(os.path.dirname(dst))
            # Remove any previous copy first, because copy_file skips a
            # destination newer than its source. A second build in the same
            # checkout would otherwise keep a configuration file that was
            # substituted for the previous build's settings.
            if os.path.exists(dst):
                os.remove(dst)
            # Follow the example of the base build_py class by not preserving
            # the mode. This ensures that the output file is read/write even if
            # the input file is read-only.
            self.copy_file(src, dst, preserve_mode=False)
            if os.path.basename(dst) == "executorch-config.cmake":
                tracer_cache_dir = _tracer_cache_dir(self)
                if tracer_cache_dir is None:
                    # An editable install reaches here because setuptools runs build_py before
                    # build_ext, so no cache exists yet. Publishing nothing would leave a developer
                    # with no find_package at all, so derive the definition from the arguments that
                    # configure the build: the setting is knowable without the cache, and absent
                    # means off, which is the project default.
                    _substitute_tracer_definition_from_args(dst)
                    _TRACER_DEFINITION_PATHS.append(dst)
                    continue
                _substitute_tracer_definition(dst, tracer_cache_dir)

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
                # And again where CMake looks when a consumer points CMAKE_PREFIX_PATH at the
                # package root, which is the ordinary way to use an installed package. CMake
                # searches <prefix>/lib/cmake/<name>, not <prefix>/share/cmake directly, so
                # without this copy the root is not a usable prefix and a consumer needs a
                # path that names this project's layout. The first location stays because the
                # existing contract uses it.
                (
                    "tools/cmake/executorch-wheel-config.cmake",
                    "lib/cmake/executorch/executorch-config.cmake",
                ),
            ]
            # The headers the package installs. Two audiences now: a custom-operator
            # build, which needs the kernel and tensor helpers, and a C++ application
            # using the shipped libraries as an SDK, which needs the documented entry
            # points as well.
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
                # Module is how the documentation tells a C++ application to load and
                # run a program. Without it the package ships the libraries to do that
                # and no way to call them, which the C++ consumer check catches.
                "extension/module/",
                # Module's constructors take unique_ptr to the runtime's allocator
                # and loader bases, whose headers already ship. These supply the
                # concrete subclasses a caller has to construct to pass one, such as
                # MallocMemoryAllocator and FileDataLoader.
                "extension/memory_allocator/",
                "extension/data_loader/",
                # The MergedDataMap and FlatTensorDataMap entry points, whose
                # implementations ship inside libexecutorch.so. The .ptd file header
                # is included too, so a caller writing or reading a .ptd by hand has
                # its declarations.
                "extension/named_data_map/merged_data_map.h",
                "extension/flat_tensor/flat_tensor_data_map.h",
                "extension/flat_tensor/serialize/flat_tensor_header.h",
                # ETDump, whose library the package ships as a component. A profiler
                # that cannot be included is a library nobody can call.
                #
                # The whole directory except the filter, which includes a regular
                # expression library the wheel does not carry and whose implementation
                # is not in the shipped library either. Publishing a header that cannot
                # be included is worse than not publishing it, because the failure
                # arrives at compile time in someone else's project.
                "devtools/etdump/etdump_flatcc.h",
                "devtools/etdump/emitter.h",
                "devtools/etdump/utils.h",
                "devtools/etdump/data_sinks/",
            ]:
                # A directory entry publishes everything under it, and a file entry publishes
                # just that file. Some directories hold headers a consumer cannot compile
                # against, so those are named individually rather than swept in.
                for src in _headers_to_install(Path(include_dir)):
                    src_to_dst.append(
                        (str(src), os.path.join("include/executorch", str(src)))
                    )
        self._copy_extra_files(src_to_dst, dst_root)

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

        if not _is_minimal_build():
            self._write_cmake_version_file(dst_root)

    def _write_cmake_version_file(self, dst_root: str) -> None:
        """Write the CMake package version file, so `find_package(executorch 1.2)` works.

        Generated rather than copied, because the version is only known here:
        version.txt gives the base and BUILD_VERSION overrides it for a nightly. A
        checked-in file would go stale the first time either changed.
        """
        template = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tools",
            "cmake",
            "executorch-wheel-config-version.cmake.in",
        )
        with open(template) as handle:
            contents = handle.read()
        # Only the numeric release part. A Python version can carry a local segment
        # such as "1.5.0+cpu" or a development suffix, and `find_package(executorch
        # 1.5.0+cpu)` is rejected by CMake as an invalid argument, so a consumer could
        # not name the version this file reports. Strip to the dotted numbers CMake
        # can compare, which is what a consumer asks for in practice.
        # Two variables with different jobs. CMake compares PACKAGE_VERSION, so it has to be the
        # numeric release and nothing else. EXECUTORCH_BUILD_VERSION is documented as the full
        # version, which is what a consumer pinning an exact build compares against, so filling it
        # from the numeric part would make that comparison pass against a different wheel.
        build_version = Version.string()
        cmake_version = re.match(r"\d+(?:\.\d+)*", build_version)
        if not cmake_version:
            # A version file claiming 0 would satisfy every version request, which is worse than
            # not building at all.
            raise RuntimeError(
                f"cannot derive a numeric CMake version from {build_version!r}; the version file "
                "would claim 0 and satisfy every version request"
            )
        contents = contents.replace("@EXECUTORCH_VERSION@", cmake_version.group(0))
        contents = contents.replace("@EXECUTORCH_BUILD_VERSION@", build_version)
        # CMake only reads a version file that sits beside the configuration file it found, so this
        # goes to both locations the configuration is installed to. Writing it to one would leave a
        # version request silently unchecked when the other location was used.
        for destination in (
            os.path.join(dst_root, "share", "cmake", "executorch-config-version.cmake"),
            os.path.join(
                dst_root,
                "lib",
                "cmake",
                "executorch",
                "executorch-config-version.cmake",
            ),
        ):
            self.mkpath(os.path.dirname(destination))
            with open(destination, "w") as handle:
                handle.write(contents)


def _headers_to_install(entry: Path):
    """The headers a copy list entry publishes, skipping any a consumer could not or should not use.

    A directory entry publishes everything under it, and a file entry publishes just that file. A header a
    consumer cannot compile is worse than an absent one, because the failure lands in their project rather
    than here, and the directory entries sweep in a few of those.

    Test directories are skipped as a whole rather than by name. They hold mocks and stubs for this
    project's own tests, nothing the wheel installs includes them, and a consumer linking a mock allocator
    or a stub platform would get behaviour no release intends. Matched on any part starting with "test", so
    a directory named testing_util counts too, which a plain equality check missed.
    """
    candidates = entry.rglob("*.h") if entry.is_dir() else [entry]
    return [
        src
        for src in candidates
        if not _is_unshippable_header(src)
        and not any(part.startswith("test") for part in src.parts[:-1])
    ]


def _is_unshippable_header(src: Path) -> bool:
    """Whether a header is on the list of ones a consumer of the wheel could not compile.

    Compared on the path ending rather than the file name. Two headers here are both called
    tensor_util.h, one a utility that many shipped headers include and one a test helper, so matching the
    bare name either kept the helper or removed the utility everything needs.
    """
    posix = src.as_posix()
    return any(posix.endswith(entry) for entry in _UNSHIPPABLE_HEADERS)


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


def _substitute_tracer_definition(path: str, cmake_cache_dir: str) -> None:
    """Fill in the tracer placeholder in an installed CMake configuration file.

    Read from the cache rather than assumed, because the option is set per platform and a builder can
    override it. A consumer compiling against the wrong setting gets a different object layout for
    the profiling scope classes, which fails silently.
    """
    cache_path = os.path.join(cmake_cache_dir, "CMakeCache.txt")
    if not os.path.exists(cache_path):
        raise RuntimeError(
            f"cannot read {cache_path}, so the tracer definition published to consumers cannot be "
            "derived from what was built"
        )
    enabled = CMakeCache(cache_path=cache_path).is_enabled(
        "EXECUTORCH_ENABLE_EVENT_TRACER"
    )
    with open(path) as handle:
        contents = handle.read()
    with open(path, "w") as handle:
        handle.write(
            contents.replace(
                "@EXECUTORCH_TRACER_DEFINITION@",
                "ET_EVENT_TRACER_ENABLED" if enabled else "",
            )
        )


# Where the first substitution wrote, so the post-build pass knows which files to correct.
_TRACER_DEFINITION_PATHS: list = []


def _substitute_tracer_definition_from_args(destination) -> None:
    """Substitute the tracer definition using the arguments that configure the build.

    Used when no CMake cache exists yet, which is the normal case for an editable install. Reading the
    same arguments CMake will read keeps the shipped config consistent with the build it describes.
    """
    # CMake accepts both -DNAME=VALUE and -D NAME=VALUE, and only the first spelling was read,
    # so the second published the tracer-off layout while the libraries carried the tracer-on
    # one. Both spellings are handled here. Absent still means off, because an editable install
    # legitimately reaches this before any cache exists and a consumer must receive a usable
    # definition rather than a raw placeholder.
    tokens = os.environ.get("CMAKE_ARGS", "").split()
    enabled = False
    for index, argument in enumerate(tokens):
        setting = None
        if argument.startswith("-DEXECUTORCH_ENABLE_EVENT_TRACER"):
            setting = argument[2:]
        elif argument == "-D" and index + 1 < len(tokens):
            setting = tokens[index + 1]
        if setting and setting.startswith("EXECUTORCH_ENABLE_EVENT_TRACER"):
            value = setting.split("=", 1)[-1].strip().upper()
            enabled = value in ("ON", "1", "TRUE", "YES")
    text = Path(destination).read_text()
    Path(destination).write_text(
        text.replace(
            "@EXECUTORCH_TRACER_DEFINITION@",
            "ET_EVENT_TRACER_ENABLED" if enabled else "",
        )
    )


def _tracer_cache_dir(command) -> Optional[str]:
    """The build directory holding CMakeCache.txt.

    Raises rather than falling back, because the value decides a compile definition that changes
    object layout. A wrong answer is silent: the consumer compiles a different version of the
    profiling scope classes and its traces come back empty with no error.
    """
    build_command = command.get_finalized_command("build")
    cache_dir = getattr(build_command, "cmake_cache_dir", None)
    if not cache_dir:
        # None rather than an error, because nothing has been built yet: an editable install runs
        # build_py before build_ext. The caller then leaves the file alone instead of guessing a
        # setting, so no config claims a tracer state that nothing verified.
        return None
    return os.path.abspath(cache_dir)


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

        # Unlike CUDA, Vulkan also needs its third-party submodules, which
        # aren't in the default checkout, along with glslc. A partial checkout
        # no-ops here rather than failing in CMake.
        vulkan_third_party = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "backends",
            "vulkan",
            "third-party",
        )
        vulkan_submodules_present = all(
            os.path.exists(os.path.join(vulkan_third_party, *parts))
            for parts in (
                ("volk", "volk.c"),
                ("Vulkan-Headers", "include", "vulkan", "vulkan.h"),
                ("VulkanMemoryAllocator", "include", "vk_mem_alloc.h"),
            )
        )
        if (
            not minimal_build
            and vulkan_submodules_present
            and install_utils.is_vulkan_available()
            and install_utils.is_cmake_option_on(
                cmake_configuration_args, "EXECUTORCH_BUILD_VULKAN", default=True
            )
        ):
            cmake_configuration_args += ["-DEXECUTORCH_BUILD_VULKAN=ON"]

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
            if cmake_cache.is_enabled("EXECUTORCH_BUILD_PYBIND"):
                cmake_build_args += ["--target", "portable_lib"]
                cmake_build_args += ["--target", "data_loader"]
                cmake_build_args += ["--target", "selective_build"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_EXTENSION_LLM_RUNNER"):
                cmake_build_args += ["--target", "_llm_runner"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_VULKAN"):
                cmake_build_args += ["--target", "vulkan_backend"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_CMSIS_NN_PYBINDS"):
                cmake_build_args += ["--target", "cmsis_nn"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_CUDA"):
                cmake_build_args += ["--target", "aoti_cuda_backend"]
                cmake_build_args += ["--target", "aoti_common_shims_slim"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_EXTENSION_MODULE"):
                cmake_build_args += ["--target", "extension_module"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_EXTENSION_TRAINING"):
                cmake_build_args += ["--target", "_training_lib"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_COREML"):
                cmake_build_args += ["--target", "executorchcoreml"]

            if cmake_cache.is_enabled("EXECUTORCH_BUILD_MLX"):
                cmake_build_args += ["--target", "mlxdelegate"]

            # Named explicitly because nothing else links it. The other shipped
            # libraries are built as dependencies of the Python extension, but a C++
            # application is the only consumer of this one, so without naming it the
            # target is generated and never built, and packaging then looks for a file
            # that does not exist.
            if cmake_cache.is_enabled("EXECUTORCH_BUILD_SHARED") and (
                cmake_cache.is_enabled("EXECUTORCH_BUILD_KERNELS_QUANTIZED")
            ):
                cmake_build_args += ["--target", "executorch_quantized_ops"]

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
                # Install the shared runtime the Python extension links, rather
                # than having the extension contain its own copy. Named without a
                # version, so a consumer's find_library(executorch) resolves it: that
                # matches libexecutorch.so and not libexecutorch.so.1. A version is
                # only useful where something upgrades the library independently of
                # what links it, which never happens inside a wheel.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/",
                    src_name="libexecutorch.so",
                    dst="executorch/lib/libexecutorch.so",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_SHARED"],
                ),
                # Install the profiler next to it, as its own library rather than
                # code fused into the Python extension, so a process has one copy of
                # it however many consumers load.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/devtools/etdump/",
                    src_name="libexecutorch_etdump.so",
                    dst="executorch/lib/libexecutorch_etdump.so",
                    # Not gated on EXECUTORCH_BUILD_DEVTOOLS. The shared build adds
                    # the devtools subdirectory itself, so the library exists
                    # whenever the shared build does. The Python extension carries a
                    # hard dependency on it, so requiring the option here left a
                    # wheel whose extension could not load at all.
                    dependent_cmake_flags=["EXECUTORCH_BUILD_SHARED"],
                ),
                # Install the shared thread pool next to it. It is a separate
                # library so that a process has one pool rather than one per
                # component that uses it.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/extension/threadpool/",
                    src_name="libexecutorch_threadpool.so",
                    dst="executorch/lib/libexecutorch_threadpool.so",
                    # The target only exists when both of its dependencies are
                    # enabled, so packaging has to require them too or a shared
                    # build with either turned off looks for a file that was
                    # never built.
                    dependent_cmake_flags=[
                        "EXECUTORCH_BUILD_SHARED",
                        "EXECUTORCH_BUILD_PTHREADPOOL",
                        "EXECUTORCH_BUILD_CPUINFO",
                    ],
                ),
                # Install the merged CPU kernels beside them, so the operators are
                # registered once per process rather than once per component.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/configurations/",
                    src_name="libexecutorch_kernels_optimized.so",
                    dst="executorch/lib/libexecutorch_kernels_optimized.so",
                    # The target is only created when the optimized kernels are
                    # enabled, so packaging has to require that too rather than
                    # looking for a file a shared build may never have produced.
                    dependent_cmake_flags=[
                        "EXECUTORCH_BUILD_SHARED",
                        "EXECUTORCH_BUILD_KERNELS_OPTIMIZED",
                    ],
                ),
                # The quantized kernels, as their own library rather than code
                # fused into the AOT-only extension beside the Python bindings.
                # A C++ application running a quantized model could not link
                # them before.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/kernels/quantized/",
                    src_name="libexecutorch_kernels_quantized.so",
                    dst="executorch/lib/libexecutorch_kernels_quantized.so",
                    dependent_cmake_flags=[
                        "EXECUTORCH_BUILD_SHARED",
                        "EXECUTORCH_BUILD_KERNELS_QUANTIZED",
                    ],
                ),
                # Install the XNNPACK delegate beside them, so a process has one
                # copy of it instead of one per component that uses it.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/backends/xnnpack/",
                    src_name="libexecutorch_backend_xnnpack.so",
                    dst="executorch/lib/libexecutorch_backend_xnnpack.so",
                    dependent_cmake_flags=[
                        "EXECUTORCH_BUILD_SHARED",
                        "EXECUTORCH_BUILD_XNNPACK",
                    ],
                ),
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
                    src="cmsis_nn.cp*" if _is_windows() else "cmsis_nn.*",
                    src_dir="backends/cortex_m/cmsis_nn-build/%BUILD_TYPE%/",
                    modpath="executorch.backends.cortex_m.library._cmsis_nn.cmsis_nn",
                    dependent_cmake_flags=[
                        "EXECUTORCH_BUILD_CMSIS_NN_PYBINDS",
                    ],
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
                # The stream helper the library above records as a dependency. It was
                # never shipped, and resolved only because the copied library still
                # carried the absolute directory it was linked in, which exists on a
                # build machine and nowhere else. Stripping that path is what made the
                # omission visible as a failed import.
                #
                # Shipped beside its consumer rather than in lib/, because that
                # directory only exists in the shared build and this has to work
                # without it. The glob covers both names the target can have: the
                # shared build renames it to advertise it as a wheel component.
                BuiltFile(
                    src_dir="%CMAKE_CACHE_DIR%/extension/cuda/%BUILD_TYPE%/",
                    src_name="*extension_cuda",
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
                    src_dir="backends/qualcomm/%BUILD_TYPE%/",
                    src=(
                        "PyQnnManagerAdaptor*.pyd"
                        if _is_windows()
                        else "PyQnnManagerAdaptor.*"
                    ),
                    modpath="executorch.backends.qualcomm.python.PyQnnManagerAdaptor",
                    dependent_cmake_flags=["EXECUTORCH_BUILD_QNN"],
                ),
            ]
        ),
    ],
    **setup_kwargs,
)
