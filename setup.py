# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import os
import re
import shutil
import subprocess
import sys
from distutils.sysconfig import get_python_lib
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.develop import develop
from setuptools.command.egg_info import egg_info
from setuptools.command.install import install

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        buck = os.environ.get("BUCK", "buck2")
        cmake_prefix_path = os.environ.get("CMAKE_PREFIX_PATH", get_python_lib())
        cmake_args = [
            "-DEXECUTORCH_BUILD_PYBIND=ON",
            "-DBUILD_SHARED_LIBS=ON",  # For flatcc
            f"-DBUCK2={buck}",
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]

        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # In this example, we pass in the version to C++. You might not need to.
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multithreads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        parallel = self.parallel if hasattr(self, "parallel") and self.parallel else "1"
        parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", parallel)
        # self.parallel is a Python 3 only way to set parallel jobs by hand
        # using -j in the build_ext call, not supported by pip or PyPA-build.
        build_args += [f"-j{parallel}"]
        print(self.build_temp)
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


def custom_command():
    src_dst_list = [
        ("schema/scalar_type.fbs", "exir/_serialize/scalar_type.fbs"),
        ("schema/program.fbs", "exir/_serialize/program.fbs"),
        (
            "sdk/bundled_program/schema/bundled_program_schema.fbs",
            "sdk/bundled_program/serialize/bundled_program_schema.fbs",
        ),
        (
            "sdk/bundled_program/schema/scalar_type.fbs",
            "sdk/bundled_program/serialize/scalar_type.fbs",
        ),
    ]
    for src, dst in src_dst_list:
        print(f"copying from {src} to {dst}")
        shutil.copyfile(src, dst)

    for _, dst in src_dst_list:
        if not os.path.isfile(dst):
            raise FileNotFoundError(
                f"Could not find {dst}, copying file from {src} fails."
            )


class CustomInstallCommand(install):
    def run(self):
        custom_command()
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        custom_command()
        develop.run(self)


class CustomEggInfoCommand(egg_info):
    def run(self):
        custom_command()
        egg_info.run(self)


cmdclass = {
    "install": CustomInstallCommand,
    "develop": CustomDevelopCommand,
    "egg_info": CustomEggInfoCommand,
}
ext_modules = None
if os.environ.get("EXECUTORCH_BUILD_PYBIND", "OFF") == "ON":
    cmdclass["build_ext"] = CMakeBuild
    ext_modules = [CMakeExtension("executorch.extension.pybindings.portable_lib")]

setup(
    package_dir={
        "executorch/backends": "backends",
        # TODO(mnachin T180504136): Do not put examples/models
        # into core pip packages. Refactor out the necessary utils
        # or core models files into a separate package.
        "executorch/examples/models": "examples/models",
        "executorch/exir": "exir",
        "executorch/extension": "extension",
        "executorch/schema": "schema",
        "executorch/sdk": "sdk",
        "executorch/sdk/bundled_program": "sdk/bundled_program",
        "executorch/util": "util",
        "serializer": "backends/arm/third-party/serialization_lib/python/serializer",
        "tosa": "backends/arm/third-party/serialization_lib/python/tosa",
    },
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
