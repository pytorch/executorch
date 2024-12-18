# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from executorch.exir.pass_base import ExportPass, NodeMetadata, ProxyValue
from executorch.exir.passes.spec_prop_pass import make_spec


class InitMutableBufferPass(ExportPass):
    def __init__(self) -> None:
        super().__init__()

    def placeholder(self, name: str, arg, meta):
        if "cache_pos" in name:
            meta["et_init_buffer"] = True

        return super().placeholder(name, arg, meta)
