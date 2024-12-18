# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.exir.pass_base import ExportPass


class CachePosToInitializedMutableBufferPass(ExportPass):
    """
    If the buffer has the name "cache_pos", such as in an kv_cache
    module with `self.register_buffer("cache_pos", torch.arange(10))`,
    mark it with a custom tag which later is used by the emitter to
    flag spec.const to True, which provides the mutable buffer with
    an initialized state.
    """

    def __init__(self) -> None:
        super().__init__()

    def placeholder(self, name: str, arg, meta):
        if "cache_pos" in name:
            meta["et_init_buffer"] = True

        return super().placeholder(name, arg, meta)
