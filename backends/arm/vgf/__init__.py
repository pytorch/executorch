# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec


def _missing_dependencies_error(missing: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "VGF backend dependencies are not installed "
        f"(missing: {missing}). Install ExecuTorch with "
        "`pip install executorch[vgf]`, or if building from source run "
        "`examples/arm/setup.sh --i-agree-to-the-contained-eula "
        "--disable-ethos-u-deps --enable-mlsdk-deps`."
    )


def _ensure_vgf_dependencies() -> None:
    required_distributions = (
        "ai_ml_emulation_layer_for_vulkan",
        "ai_ml_sdk_model_converter",
        "ai_ml_sdk_vgf_library",
    )
    required_modules = {
        "tosa_serializer": "tosa-tools",
    }
    missing_packages = []
    for package_name in required_distributions:
        try:
            version(package_name)
        except PackageNotFoundError:
            missing_packages.append(package_name)

    for module_name, package_name in required_modules.items():
        try:
            if find_spec(module_name) is None:
                missing_packages.append(package_name)
        except (ImportError, ValueError):
            missing_packages.append(package_name)

    if missing_packages:
        raise _missing_dependencies_error(", ".join(missing_packages))


_ensure_vgf_dependencies()

from .backend import VgfBackend  # noqa: F401
from .compile_spec import VgfCompileSpec  # noqa: F401
from .partitioner import VgfPartitioner  # noqa: F401

__all__ = ["VgfBackend", "VgfPartitioner", "VgfCompileSpec"]
