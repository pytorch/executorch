# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List

from executorch.exir.pass_base import ExportPass


class InitializedMutableBufferPass(ExportPass):
    """
    If a buffer has a name that within a specified list, set meta["et_init_buffer"]
    to True, which provides the mutable buffer with an initialized state.

    As an example, a module with `self.register_buffer("cache_pos", torch.arange(10))`
    when patterns = ["cache_pos"] would have its initial state set instead of being
    left uninitialized by default.
    """

    def __init__(self, patterns: List[str]) -> None:
        super().__init__()
        self.patterns = patterns

    def placeholder(self, name: str, arg, meta):
        for pattern in self.patterns:
            if pattern in name:
                meta["et_init_buffer"] = True

        return super().placeholder(name, arg, meta)
