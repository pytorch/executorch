# Copyright 2024-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Expose quantizer APIs and load optional quantized kernels.

Import the public quantizer classes and configuration helpers for Arm backends.
Attempt to load portable and quantized libraries; fall back to a log message if
unavailable.

"""

from .quantization_config import QuantizationConfig  # noqa  # usort: skip

# Used in tests
from .arm_quantizer_utils import is_annotated  # noqa

# Lazily import heavy quantizer classes to avoid circular imports with
# Cortex-M quantization configs.
_LAZY_EXPORTS = {
    "EthosUQuantizer": "executorch.backends.arm.quantizer.arm_quantizer",
    "get_symmetric_a16w8_quantization_config": "executorch.backends.arm.quantizer.arm_quantizer",
    "get_symmetric_quantization_config": "executorch.backends.arm.quantizer.arm_quantizer",
    "TOSAQuantizer": "executorch.backends.arm.quantizer.arm_quantizer",
    "VgfQuantizer": "executorch.backends.arm.quantizer.arm_quantizer",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(_LAZY_EXPORTS[name])
    return getattr(module, name)


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))


# Load quantized ops library.
try:
    import executorch.extension.pybindings.portable_lib
    import executorch.kernels.quantized  # noqa
except:
    import logging

    logging.info(
        "Failed to load portable_lib and quantized_aot_lib. To run quantized kernels AOT, either build "
        "Executorch with pybindings, or load your own custom built op library using torch.ops.load_library."
    )
    del logging
