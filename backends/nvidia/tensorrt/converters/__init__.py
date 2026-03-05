# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TensorRT converters for ExecuTorch operations."""

# Import converters to trigger registration via @converter decorator
from executorch.backends.nvidia.tensorrt.converters import add  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import div  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import mm  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import mul  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import relu  # noqa: F401
from executorch.backends.nvidia.tensorrt.converters import sub  # noqa: F401
