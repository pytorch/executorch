# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared torch.library handle for the ``executorch_cuda`` op namespace.

``int4_dispatch``, ``int6_dispatch`` and ``int8_dispatch`` all register custom
ops into the same ``executorch_cuda`` namespace, so they must share a single
``DEF`` library instance — PyTorch allows only one ``DEF`` per namespace per
process.
"""

from torch.library import Library

lib = Library("executorch_cuda", "DEF")
