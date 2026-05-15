#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

# Trigger op/pattern handler registration.
# ops.py and patterns.py use @REGISTRY.register() decorators at import time.
# This must happen after REGISTRY is defined (in op_registry.py).
from executorch.backends.mlx import ops, patterns  # noqa: F401
from executorch.backends.mlx.builder.op_registry import REGISTRY  # noqa: F401
from executorch.backends.mlx.builder.program_builder import (  # noqa: F401
    MLXProgramBuilder,
)
