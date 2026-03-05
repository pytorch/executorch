# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT converters for ExecuTorch operations."""

# Import converters to trigger registration via @converter decorator
from executorch.backends.nvidia.tensorrt.converters import activations  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import add  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import addmm  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import batch_norm  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import clamp  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import concat  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import conv2d  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import dim_order_ops  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import div  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import embedding  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import expand  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import getitem  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import layer_norm  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import linear  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import mm  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import mul  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import permute_copy  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import pixel_shuffle  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import pooling  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import reduction  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import relu  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import reshape  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import sub  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import upsample  # noqa: F401


def clear_converter_weight_storage() -> None:
    """Clear weight storage to free memory after engine build."""
    conv2d.clear_weight_storage()
    batch_norm.clear_weight_storage()
    linear.clear_weight_storage()
