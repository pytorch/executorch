# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

from importlib.util import find_spec


def _missing_dependencies_error(missing: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "Ethos-U backend dependencies are not installed "
        f"(missing: {missing}). Install ExecuTorch with "
        "`pip install executorch[ethos_u]`, or if building from source run "
        "`examples/arm/setup.sh --i-agree-to-the-contained-eula`."
    )


def _ensure_ethos_u_dependencies() -> None:
    required_modules = {
        "ethosu.vela": "ethos-u-vela",
        "tosa_serializer": "tosa-tools",
    }
    missing_packages = []
    for module_name, package_name in required_modules.items():
        try:
            if find_spec(module_name) is None:
                missing_packages.append(package_name)
        except (ImportError, ValueError):
            missing_packages.append(package_name)

    if missing_packages:
        raise _missing_dependencies_error(", ".join(missing_packages))


_ensure_ethos_u_dependencies()

from .backend import EthosUBackend  # noqa: F401
from .compile_spec import EthosUCompileSpec  # noqa: F401
from .partitioner import EthosUPartitioner  # noqa: F401

__all__ = ["EthosUBackend", "EthosUPartitioner", "EthosUCompileSpec"]
