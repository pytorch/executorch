import logging
import os
import sys
from typing import Dict, List

import torch
from torch_tensorrt._version import (  # noqa: F401
    __cuda_version__,
    __version__,
)

from packaging import version

if sys.version_info < (3,):
    raise Exception(
        "Python 2 has reached end-of-life and is not supported by Torch-TensorRT"
    )

_LOGGER = logging.getLogger(__name__)

# --- fbsource patch: skip _TensorRTProxyModule (needs libnvinfer.so at import) ---
# The _TensorRTProxyModule tries to import tensorrt which needs libnvinfer.so.
# In fbsource, tensorrt is provided via //deeplearning/trt/python:py_tensorrt
# which already handles the native library loading through buck.
try:
    from . import _TensorRTProxyModule  # noqa: F401
except Exception:
    _LOGGER.debug("_TensorRTProxyModule not available (expected in fbsource buck builds)")


# --- fbsource patch: skip _register_with_torch (needs libtorchtrt.so) ---
# libtorchtrt.so provides TorchScript IR support and native runtime ops.
# The dynamo IR compilation path is pure Python and doesn't need it.
def _register_with_torch() -> None:
    trtorch_dir = os.path.dirname(__file__)
    linked_file = os.path.join(
        "lib", ("torchtrt.dll" if sys.platform.startswith("win") else "libtorchtrt.so")
    )
    linked_file_runtime = os.path.join(
        "lib",
        (
            "torchtrt_runtime.dll"
            if sys.platform.startswith("win")
            else "libtorchtrt_runtime.so"
        ),
    )
    linked_file_full_path = os.path.join(trtorch_dir, linked_file)
    linked_file_runtime_full_path = os.path.join(trtorch_dir, linked_file_runtime)

    if os.path.isfile(linked_file_full_path):
        try:
            torch.ops.load_library(linked_file_full_path)
        except OSError:
            _LOGGER.debug("libtorchtrt.so not loadable; TorchScript IR disabled")

    elif os.path.isfile(linked_file_runtime_full_path):
        try:
            torch.ops.load_library(linked_file_runtime_full_path)
        except OSError:
            _LOGGER.debug("libtorchtrt_runtime.so not loadable")


from torch_tensorrt._features import ENABLED_FEATURES, _enabled_features_str

_LOGGER.debug(_enabled_features_str())

_register_with_torch()

from torch_tensorrt._Device import Device  # noqa: F401
from torch_tensorrt._enums import (  # noqa: F401
    DeviceType,
    EngineCapability,
    Platform,
    dtype,
    memory_format,
)
from torch_tensorrt._Input import Input  # noqa: F401
from torch_tensorrt.runtime import *  # noqa: F403

if ENABLED_FEATURES.torchscript_frontend:
    from torch_tensorrt import ts

if ENABLED_FEATURES.fx_frontend:
    from torch_tensorrt import fx

if ENABLED_FEATURES.dynamo_frontend:
    from torch_tensorrt.dynamo import backend  # noqa: F401
    from torch_tensorrt import dynamo  # noqa: F401

from torch_tensorrt._compile import *  # noqa: F403
from torch_tensorrt.dynamo.runtime._MutableTorchTensorRTModule import (
    MutableTorchTensorRTModule,
)
