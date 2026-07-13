# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized-weight F.linear dispatch for CUDA — eager / export trace time.

Importing this package overrides the F.linear dispatch of torchao quantized
weight tensors so that torch.export traces through ExecuTorch's custom ops and
dequant logic instead of torchao's defaults. It registers:

  * INT4 (``CudaCoalescedInt4Tensor``)  → ``executorch_cuda::int4_plain_mm``
  * INT5 (``CudaDp4aPlanarInt5Tensor``) → ``executorch_cuda::int5_plain_mm``
  * INT6 (``CudaDp4aPlanarInt6Tensor``) → ``executorch_cuda::int6_plain_mm``
  * INT8 (``IntxUnpackedToInt8Tensor``) → ``executorch_cuda::int8_plain_mm``

See ``int4_dispatch``, ``int5_dispatch``, ``int6_dispatch`` and ``int8_dispatch``
for the per-dtype details.

Import this package before using nn.Linear with quantized weights::

    import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401
"""

from executorch.backends.cuda.quantize_op_dispatch import (  # noqa: F401
    int4_dispatch,
    int5_dispatch,
    int6_dispatch,
    int8_dispatch,
)
