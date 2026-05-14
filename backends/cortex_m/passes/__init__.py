# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib.util import find_spec


def _missing_dependencies_error(missing: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        "Cortex-M backend dependencies are not installed "
        f"(missing: {missing}). Install ExecuTorch with "
        "`pip install executorch[cortex_m]`, or if building from source run "
        "`examples/arm/setup.sh --i-agree-to-the-contained-eula`."
    )


def _ensure_cortex_m_dependencies() -> None:
    required_modules = {
        "cmsis_nn": "cmsis_nn",
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


_ensure_cortex_m_dependencies()

from .activation_fusion_pass import ActivationFusionPass  # noqa
from .clamp_hardswish_pass import ClampHardswishPass  # noqa
from .convert_to_cortex_m_pass import ConvertToCortexMPass  # noqa
from .cortex_m_pass import CortexMPass  # noqa
from .decompose_hardswish_pass import DecomposeHardswishPass  # noqa
from .decompose_mean_pass import DecomposeMeanPass  # noqa
from .quantized_clamp_activation_pass import QuantizedClampActivationPass  # noqa
from .quantized_op_fusion_pass import QuantizedOpFusionPass  # noqa
from .replace_quant_nodes_pass import ReplaceQuantNodesPass  # noqa
from .cortex_m_pass_manager import CortexMPassManager  # noqa  # usort: skip
