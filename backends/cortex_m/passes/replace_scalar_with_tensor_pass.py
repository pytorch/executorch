# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.transforms.replace_scalar_with_tensor import (
    ReplaceScalarWithTensorArgPass,
)

from .cortex_m_quantized_pass import CortexMQuantizedPass


class CortexMReplaceScalarWithTensorArgPass(
    ReplaceScalarWithTensorArgPass,
    CortexMQuantizedPass,
):
    """Scalar-to-tensor replacement with Cortex-M quantized retracing."""
