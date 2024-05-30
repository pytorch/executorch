import os

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent.parent.parent.parent
PYBIND11_DIR_PATH = REPO_ROOT / "third-party" / "pybind11"
sys.path.append(str(PYBIND11_DIR_PATH.absolute()))

from pybind11.setup_helpers import build_ext, Pybind11Extension
from setuptools import setup

__version__ = "0.0.1"

cxx_std = int(os.environ.get("CMAKE_CXX_STANDARD", "17"))

ext_modules = [
    Pybind11Extension(
        "executorchcoreml",
        [
            "../util/json_util.cpp",
            "inmemory_filesystem.cpp",
            "inmemory_filesystem_py.cpp",
            "inmemory_filesystem_utils.cpp",
            "memory_buffer.cpp",
            "memory_stream.cpp",
            "reversed_memory_stream.cpp",
        ],
        define_macros=[("VERSION_INFO", __version__)],
        cxx_std=cxx_std,
        extra_compile_args=["-mmacosx-version-min=10.15", "-g"],
        include_dirs=[
            "../../third-party/nlohmann_json/single_include",
            ".",
            "../util",
        ],
    ),
]

setup(
    name="executorchcoreml",
    version=__version__,
    description="CoreML extension for executorch",
    long_description="",
    author="Apple Inc.",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
)
