# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.native.passes import get_default_passes
from executorch.exir import EdgeCompileConfig

__all__ = [
    "get_default_compile_config",
    "get_default_passes",
]


# pyre-ignore[11]: EdgeCompileConfig is defined in a pyre-unsafe exir module.
def get_default_compile_config() -> EdgeCompileConfig:
    """EdgeCompileConfig shared by the native export and ET lowering paths.

    IR validity checking is off and dim order is skipped: the native backend's
    passes emit non-core ops and it serializes dim_order itself.
    """
    return EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True)
